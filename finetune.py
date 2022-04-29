import numpy as np
import math
import os, glob , inspect
from genericpath import exists
from datetime import datetime
import pandas as pd 

import torch
import torch.nn as nn
from torch.utils.data import  DataLoader

from sklearn.metrics import  confusion_matrix, accuracy_score, classification_report
from data_loader import data_loader

'''
finetune model for attention augmentation 
..\\Results\\finetune\\ [DATA NAME]\\saved_models \\ FE : path of finetuned model wegith. 
..\\Results\\ finetune\\ [DATA NAME] \\ result_tex \\ : path of result file. 
..\\Results\\ finetune\\ [DATA NAME] \\ result_txt \\ : path of result file. 

'''
class Attention(nn.Module):
    """  attention Layer"""
    def __init__(self):
        super(Attention, self).__init__()
        
        self.mask_ratio = 0.1
        self.dropout = nn.Dropout(self.mask_ratio)
        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self, query, key, value, mask):
        """
            Args :
                query : input feature maps(i.e. conv1d output)
                key : conv1d input value 
                value: conv1d input value  (same with key)

            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        d_k = query.size(-1) # 128, 488
        #scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k) # (batch , 128, 128)
        scores = torch.matmul(query.transpose(-2,-1), key) / math.sqrt(d_k) # (batch , 128, 128)
        p_attn = self.softmax(scores)
        if mask:
            p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value.transpose(-2,-1))
        return out.transpose(-2,-1), p_attn  # [Batch, 128, 483]


class Front_end(nn.Module):

    def __init__(self, input_dim=2, input_size=3000):
        super(Front_end, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size

        self.conv1d = nn.Sequential(
            nn.Conv1d(self.input_dim , 64, kernel_size=50, stride=6, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )

        self.conv1d_2 = nn.Sequential(

            nn.utils.spectral_norm(nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=1)),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 488, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )

        self.fc_local = nn.Sequential(
            nn.Linear(128*71, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        self.attn = Attention()

    def forward(self, input, mask, local): #[20, 1, 3000]
        x = self.conv1d(input) # [B,128,488] , [B,128,71]
        x_2 = self.conv1d_2(x) # [B,128,483] , [B,128,66]
        if local:
            x, attn_prob = self.attn(query=x, key=x_2, value=x_2, mask=mask) #[B, 128,71]
            x = x.reshape(-1, 128*71)
            a = self.fc_local(x) # [B,1024]
        else:
            x, attn_prob = self.attn(query=x, key=x_2, value=x_2, mask=mask) 
            out = x.reshape(-1, 128 * 488) 
            a = self.fc(out) #[B, 1024]
        return a

## Q 
#Architecutre of AUXALIARY MODEL (Q)
class Latent_predictor(nn.Module):
    # Module which reconstructs the latent codes from the fake images
    def __init__(self, len_discrete_code = 5):
        super(Latent_predictor, self).__init__()

        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)

        self.fc = nn.Sequential(
            nn.Linear(1024, 128), 
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 =nn.Linear(128, self.len_discrete_code)
        

    def forward(self, input):

        a = self.fc(input) #[20, 128]
        b = self.fc1(a)  #[20, 5]
        return a,b

class fcn_model(nn.Module):
    def __init__(self):
        super(fcn_model,self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128,5)
        )
        
    def forward(self, input):
        output = self.fc(input)
        return output



class Final_predict(object):
    def __init__(self) :

        ## dir 
        self.root_dir = '..\\data\\'
        self.eeg_path = 'sleep_edfx\\EEG_test\\'  # options  [ISRUC\\EEG, sleep_edfx\\EEG_test]
        self.eog_path = 'sleep_edfx\\EOG_test\\'  # options  [ISRUC\\EOG, sleep_edfx\\EOG_test\\] 
        
        self.n_epochs = 10
        self.batch_size = 64
        self.kfold = False
        self.weight_name = "sleep_edfx"
        self.label_ratio = 0.2 # ratio of label you want to use. 
        self.is_save = True
        self.info  = "attention based augmentation"
        self.data_name = self.eeg_path.split('\\')[0]
        self.f_name = str(inspect.getfile(inspect.currentframe())[:-3])

        if self.data_name == 'sleep_edfx': # fineune model with sleep-edfx data 
            self.PATH1 = glob.glob(os.path.join('.\\model_weight\\',"netFE100_si*.pth" )) 
            self.PATH2 = glob.glob(os.path.join('.\\model_weight\\',"netQ100_si*.pth" ))  
        
        # Fine tune our model  with other dataset (e.g. ISRUC)
        elif self.kfold: # fine tune our model with k fold weight. 
            self.PATH1 = glob.glob(os.path.join(f'..\\Results\\finetune_simsiam\\{self.weight_name}\\saved_models\\netFE\\',"netFE5*.pth" )) 
            self.PATH2 = glob.glob(os.path.join(f'..\\Results\\finetune_simsiam\\{self.weight_name}\\saved_models\\netQ\\',"netQ5*.pth" ))
        
        else: # fine tune our model with few label weight. 
            self.PATH1 = glob.glob(os.path.join(f'..\\Results\\{self.f_name}\\{self.weight_name}\\saved_models\\netFE\\',"netFE5*.pth" )) 
            self.PATH2 = glob.glob(os.path.join(f'..\\Results\\{self.f_name}\\{self.weight_name}\\saved_models\\netQ\\',"netQ5*.pth" ))
    
        self.PATH1  = self.PATH1.pop()
        self.PATH2  = self.PATH2.pop()
        print('='*10)
        print("model version")
        print(self.PATH1)
        print(self.PATH2)
        print('='*10)

        # dir for save model 
        self.data_name = self.eeg_path.split('\\')[0] 
        self.save_dir = os.path.join("..\\Results\\", self.f_name+"\\"+self.data_name) 
        self.model_path = os.path.join(self.save_dir, "saved_models") 
        self.result_path = os.path.join(self.save_dir, "result_txt\\") 
        self.result_path_tex = os.path.join(self.save_dir, "result_tex\\" )  

        # save info. 
        self.exp_date = datetime.now().strftime("_%m%d_%H_%M_%S") 
        self.model_name ='FE'+self.PATH1.split('\\')[-1][5:7] 
        print('='*20)
        print("result path")
        print(self.result_path)
        print(self.result_path_tex)
        print('='*20)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            os.makedirs(self.model_path) 
            os.makedirs(self.result_path_tex)
            os.makedirs(self.result_path)
        
        ## dataloader  
        self.data_loader = data_loader.dataloader(self.root_dir, self.eeg_path, self.eog_path, batch_size = self.batch_size, normalization = True)
        torch.manual_seed(0)

        # Split tarin and test dataset 
        self.train_set, self.test_set = torch.utils.data.random_split(self.data_loader.dataset,[math.floor(self.data_loader.dataset.__len__() * self.label_ratio), math.ceil(self.data_loader.dataset.__len__() * (1-self.label_ratio))])
        self.train_loader = DataLoader(dataset = self.train_set, 
                                        batch_size = self.batch_size)
        self.test_loader = DataLoader(dataset = self.test_set, 
                                    batch_size = self.batch_size)
        
        self.device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")


        ## model init  
        self.FE_model = Front_end()
        self.Q_model = Latent_predictor()
        self.fcn = fcn_model().to(self.device)
        

        ## load trained weight.  
        self.FE_model.load_state_dict(torch.load(self.PATH1))
        self.FE_model.eval()
        self.Q_model.load_state_dict(torch.load(self.PATH2)) 
        self.criterion  = nn.CrossEntropyLoss().cuda()

        self.FE_model = self.FE_model.to(self.device)
        self.Q_model = self.Q_model.to(self.device)
    
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam([{'params':self.Q_model.parameters()}], lr = self.learning_rate, betas=(0.9, 0.98))
       

        # gradient backward flow -> False
        for param in self.FE_model.parameters():
            param.requires_grad = False

        for param in self.Q_model.parameters():
            param.requires_grad = True
            self.model_name = 'Q'+self.PATH1.split('\\')[2][9:11]

    def train(self):
        print('Final Predict Training Start!')
        print(len(self.train_loader.dataset))
        print(len(self.test_loader.dataset))

        n_total_steps = len(self.train_loader)

        #train
        for epoch in range(self.n_epochs):
            for iter, (x_,_,_, y_) in enumerate(self.train_loader):

                x_ = x_.type(torch.FloatTensor) # type casting    
                y_ = y_.type(torch.LongTensor) # type casting            
                x_ = x_.to(self.device)
                y_ = y_.to(self.device)

                #forward
                self.optimizer.zero_grad()
                
                x = self.FE_model(x_, mask=False, local=False)
                latent, label = self.Q_model(x)
                
                loss = self.criterion(label, y_)

                #backward
                #loss.backward(retain_graph = True)
                loss.backward()
                self.optimizer.step()

                if (iter+1) % 10 == 0:
                    print(f'epoch {epoch+1} / {self.n_epochs}, step {iter+1}/{n_total_steps}, loss = {loss.item():.4f}')

            # data_name_FE/Q_version_DATE_H_M_S
            #save_name = self.data_name + "_" + self.model_name + "_" + self.exp_num + self.exp_date+ ".pkl"
            #save_dir = "Results"
            if not os.path.exists(os.path.join(self.model_path, 'netQ')):
                os.makedirs(os.path.join(self.model_path, 'netFE'))
                os.makedirs(os.path.join(self.model_path, 'netQ'))
            if self.is_save:
                torch.save(self.FE_model.state_dict(), os.path.join(self.model_path, 'netFE\\netFE{}_{}.pth'.format((epoch+1),self.exp_date)))
                torch.save(self.Q_model.state_dict(), os.path.join(self.model_path, 'netQ\\netQ{}_{}.pth'.format((epoch+1),self.exp_date)))


            
    def test(self):
        with torch.no_grad():
            self.Q_model.eval()
            predlist=torch.zeros(0,dtype=torch.long, device='cpu')
            lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

            for _, (x_,_, _, label) in enumerate(self.test_loader):
                x_ = x_.type(torch.FloatTensor) # type casting
                label = label.type(torch.FloatTensor) # type casting
                
                x_ = x_.to(self.device)
                label = label.to(self.device)
  
                x = self.FE_model(x_, mask=False, local=False)
                latent, pred = self.Q_model(x)
                
                
                _, predictions = torch.max(pred, 1)

                predlist = torch.cat([predlist, predictions.view(-1).cpu()])
                lbllist = torch.cat([lbllist, label.view(-1).cpu()])
            
            conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
            class_accuracy = 100*conf_mat.diagonal()/conf_mat.sum(1)
            acc = accuracy_score(lbllist, predlist)
            report = classification_report(lbllist, predlist, target_names = ["W", "N1", "N2", "N3", "REM"], output_dict=True)
          
            # save as a latex file 
            df = pd.DataFrame(report).transpose()
            save_name_tex = self.model_name + "_" + self.exp_date+ ".tex" 
            
            if self.is_save:
                df.to_latex(os.path.join(self.save_dir, save_name_tex))


            print('confusion matrix\n', conf_mat)
            print('')
            print('class accuracy\n', class_accuracy)
            print('')
            print('final accuracy\n', acc)
            print(classification_report(lbllist, predlist, target_names = ["W", "N1", "N2", "N3", "REM"]))

            # save as a txt file 
            save_name = self.model_name + "_" + self.exp_date+ ".txt" 
            
            if self.is_save:
                with open(os.path.join(self.result_path, save_name),"w") as txt_file:
                    print(self.info, file = txt_file)
                    print("weight path: ", self.PATH1, file=txt_file)
                    print("saved model path: ", self.model_path, file=txt_file)
                    print("label ratio:{}%".format(self.label_ratio), file=txt_file)
                    print('-------------------'*3,file=txt_file)
                    
                    print(classification_report(lbllist, predlist, target_names = ["W", "N1", "N2", "N3", "REM"])+
                    "\n\n batch_size:{} \n\n epoch:{}".format(self.batch_size, self.n_epochs), file = txt_file)
                    print('\n\n confusion matrix \n', conf_mat, file = txt_file)
                    print('\n\n final accuracy',acc, file = txt_file)
                    txt_file.close()

def main():
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")


    output = Final_predict()
    output.train()
    output.test()

if __name__ =='__main__':
    main()



