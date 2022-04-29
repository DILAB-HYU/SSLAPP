

import utils, time, os, pickle
from datetime import datetime
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import data_loader
from torch.autograd import Variable 
from GAN import Attention, Attention_Random, Generator, Discriminator, Front_end, Latent_predictor, Latent_pred




class SSLLAP(object):
    def __init__(self, args):
        
        ########## PARAMETER SETTING ##########
        # training parameters
        self.epoch = args.epoch
        self.mytemp = args.mytemp
        self.klwt = args.klwt # kl weight 
        self.lambda_g = args.lambda_g #global loss weight
        self.lambda_l = args.lambda_l #local loss weight
        self.segment = args.segment # segment 
        self.seg_length = 3000//self.segment  #one segment length
        self.batch_size = args.batch_size
        self.gpu_mode = args.gpu_mode

        # data dir 
        self.root_dir = args.root_dir
        self.eeg_path = args.eeg_path
        self.eog_path = args.eog_path  
        
        # save info 
        self.exp_date = datetime.now().strftime("_%m%d_%H_%M_%S") 
        self.model_name = "test_v3"  
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.result_dir_eeg = os.path.join(self.result_dir, args.result_dir_eeg)
        self.result_dir_eog = os.path.join(self.result_dir, args.result_dir_eog)

        # parameter for generative model 
        self.z_dim = 62 # number of noise 
        self.len_discrete_code = 5   # number of classes(i.e. label)
        self.sample_num = 100
        temp = torch.tensor(self.len_discrete_code * [float(1)/ self.len_discrete_code]).cuda()  
        self.prior_parameters = Variable(temp, requires_grad = True)

        ########## Data Load ##########
        self.data_loader = data_loader.dataloader(self.root_dir, self.eeg_path, self.eog_path, self.batch_size, normalization=True)
        
        print ("Totall length of dataloader", len(self.data_loader))
        data = self.data_loader.__iter__().__next__()[0]

        ########## NETWORK INIT ##########
        self.G = Generator(input_dim = self.z_dim, output_dim = data.shape[1], len_discrete_code = self.len_discrete_code)
        self.FE = Front_end(input_dim = data.shape[1],  segment = self.segment)
        self.D = Discriminator(output_dim=1)
        self.Q = Latent_predictor(len_discrete_code = self.len_discrete_code)
        self.Q_pred = Latent_pred()
        self.G_optimizer = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()}, {'params':self.prior_parameters}, {'params':self.Q_pred.parameters()}], lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.FE.cuda()
            self.Q.cuda()
            self.Q_pred.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.BCE_logits = nn.BCEWithLogitsLoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
            self.MSE_loss = nn.MSELoss().cuda()
            

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # sample z for GAN 
        self.sample_z_ = torch.randn((self.sample_num, self.z_dim)) 
        temp = torch.zeros((self.len_discrete_code, 1)) 
        for i in range(self.len_discrete_code):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.sample_num):
                temp_y[i] = temp_y[i] + (i / (self.sample_num/self.len_discrete_code)) 
        
        self.sample_y_ = torch.zeros((self.sample_num, self.len_discrete_code)).scatter_(1, temp_y.type(torch.LongTensor), 1)

        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = \
                self.sample_z_.cuda(), self.sample_y_.cuda()
                

    def sample_gumbel(self, shape, eps = 1e-20):
        u = torch.FloatTensor(shape, self.len_discrete_code).cuda().uniform_(0, 1)
        return -torch.log(-torch.log(u + eps) + eps)

    def gumbel_softmax_sample(self, logits, temp, batch_size):
        y = logits + self.sample_gumbel(batch_size)
        return F.softmax( y / temp)

    def approx_latent(self, params):
        params = F.softmax(params)
        log_params = torch.log(params)
        c = self.gumbel_softmax_sample(log_params, temp = 0.1, batch_size = self.batch_size) 
        return c


    def slice_segment(self, x, seg_length):
        sequence = [x[:,:,i:i + seg_length] for i in range(0, x.size(dim=2), seg_length)]
        return sequence
    
    def func(self, seg):
        a,b = self.FE(seg,mask=True,local=True), self.FE(seg,mask=True,local=True)
        (A_pred,_) , (B_pred,_) = self.Q(a), self.Q(b)
        A_proj, B_proj = self.Q_pred(A_pred), self.Q_pred(B_pred)
        return A_proj, A_pred, B_proj, B_pred

    def kl(self, A_proj, A_pred, B_proj, B_pred):
        loss = (-F.cosine_similarity(A_proj, B_pred.detach(), dim=-1).mean() / 2) + (-F.cosine_similarity(B_proj, A_pred.detach(), dim=-1).mean() / 2)
        l_loss = loss
        return l_loss

    def train(self):

        self.train_hist = defaultdict(list)
        self.train_hist = {'D_loss':[],'G_loss':[], 'info_loss':[], 'segment_loss':[], 
                           'kl_loss':[], 'per_epoch_time':[], 'total_time':[]}

        self.prior_denominator = 3 
        
        # label initialization: real :1 , fake: 0
        self.y_real_f, self.y_real_, self.y_fake_ =  torch.ones(self.batch_size, 1, device = "cuda"), torch.full((self.batch_size, 1),0.9, device = "cuda"), torch.zeros(self.batch_size, 1, device = "cuda") 
        
        self.D.train()
        
        print('start training')
        start = torch.cuda.Event(enable_timing=True) 
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        for epoch in range(self.epoch):
            
            self.G.train()
            
            per_epoch_start = torch.cuda.Event(enable_timing=True) 
            per_epoch_end = torch.cuda.Event(enable_timing=True)
            per_epoch_start.record()
        
            for iter, (x_, _, _, _) in enumerate(self.data_loader):
                
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                
                # sample noise for generate fake data 
                z_ = torch.randn((self.batch_size, self.z_dim), device = "cuda")
                
                # sample latent code 
                y_disc_= self.approx_latent(self.prior_parameters)
                
                x_ = x_.type(torch.FloatTensor) # type casting
                
                if self.gpu_mode:
                    x_  = x_.cuda()

                
                ########## UPDATE D network ##########
                self.D_optimizer.zero_grad()

                # real part
                real_intm = self.FE(x_,mask=False, local = False)
                real_intm_ = self.FE(x_,mask=True, local = False)
                real_intm_aux = self.FE(x_, mask = True, local = False)
                real_logits = self.D(real_intm)
                D_real_loss = self.BCE_logits(real_logits, self.y_real_)

                # fake part
                fx = self.G(z_, y_disc_) 
                fake_intm_tmp = self.FE(fx.detach(), mask = False, local = False) 
                fake_logits_tmp = self.D(fake_intm_tmp)
                D_fake_loss = self.BCE_logits(fake_logits_tmp, self.y_fake_)
                D_loss = D_real_loss + D_fake_loss

                D_loss.backward()
                self.D_optimizer.step()

                ##############################################################
                
                ########## UPDATE G network ##########
                self.G_optimizer.zero_grad()

                fake_intm = self.FE(fx,mask = False, local = False)
                fake_logits = self.D(fake_intm)
                G_fake_loss = self.BCE_logits(fake_logits, self.y_real_f)
               

                # information loss
                _,c_pred = self.Q(fake_intm) #[20, 5]
                info_loss = self.CE_loss(c_pred, torch.max(y_disc_, 1)[1]) 
            

                ######################local, global loss ##################
                # local loss
                kl_local = 0
                for segment in self.slice_segment(x_,self.seg_length):
                    A_proj, A_pred, B_proj, B_pred = self.func(segment)
                    kl_local += self.kl(A_proj, A_pred, B_proj, B_pred)
                kl_local = kl_local / self.segment

                # global loss
                (real_pred, _), (real_pred_pos, _ ) = self.Q(real_intm_.detach()), self.Q(real_intm_aux.detach()) # z1 - encoder & augmentation z2 - encoder & augmentation 
                real_proj, real_proj_pos = self.Q_pred(real_pred), self.Q_pred(real_pred_pos) # p1 p2 
                kl_global = self.kl(real_proj,real_pred,real_proj_pos,real_pred_pos)

                kl_loss = kl_global*self.lambda_g + kl_local*self.lambda_l

                G_loss = G_fake_loss + info_loss + self.klwt*kl_loss 

                G_loss.backward()
                self.G_optimizer.step()

                # loss plot 
                self.train_hist['D_loss'].append(D_loss.item())
                self.train_hist['G_loss'].append(G_fake_loss.item())
                self.train_hist['info_loss'].append(info_loss.item())
                self.train_hist['kl_loss'].append(kl_loss.item())

                
                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.4f, G_loss: %.4f, Info_loss: %4f, kl_global: %.4f, kl_local: %.4f " %
                            ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_fake_loss.item(), info_loss.item(), kl_global.item(), kl_local.item())) 
                
            per_epoch_end.record()   
            torch.cuda.synchronize()
            self.train_hist['per_epoch_time'].append(per_epoch_start.elapsed_time(per_epoch_end))

            if (epoch+1) % 10 == 0:
                torch.save(self.G.state_dict(), os.path.join('{}\\saved_models\\netG'.format(self.save_dir), 'netG{}_{}.pth'.format((epoch+1), self.exp_date)))
                torch.save(self.D.state_dict(), os.path.join('{}\\saved_models\\netD'.format(self.save_dir), 'netD{}_{}.pth'.format((epoch+1), self.exp_date)))
                torch.save(self.FE.state_dict(), os.path.join('{}\\saved_models\\netFE'.format(self.save_dir), 'netFE{}_{}.pth'.format((epoch+1), self.exp_date)))
                torch.save(self.Q.state_dict(), os.path.join('{}\\saved_models\\netQ'.format(self.save_dir), 'netQ{}_{}.pth'.format((epoch+1), self.exp_date)))
       
        end.record()   
        torch.cuda.synchronize()      
        self.train_hist['total_time'].append(start.elapsed_time(end)) 
        
        print("Avg of one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save() 

        self.loss_plot(self.train_hist, os.path.join('.\\{}\\visualize\\'.format(self.save_dir), self.model_name), self.model_name, self.exp_date)

  
    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))
    
        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    def loss_plot(self, hist, path='Train_hist.png', model_name='', exp_date =''):
        x = range(len(hist['D_loss']))

        y1 = hist['D_loss']
        y2 = hist['G_loss']
        y3 = hist['info_loss']
        y4 = hist['kl_loss']

        plt.plot(x, y1, label='D_loss') 
        plt.plot(x, y2, label='G_loss')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(path, model_name + '_D&G_loss' + exp_date + '.png')
        plt.savefig(path)

        plt.plot(x, y3, label='info_loss')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(path, model_name + '_info_loss' + exp_date + '.png')
        plt.savefig(path)

        plt.plot(x, y4, label='kl_loss')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(path, model_name + '_kl_loss_loss' + exp_date + '.png')
        plt.savefig(path)

