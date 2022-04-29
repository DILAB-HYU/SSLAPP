'''
Note: Generative model baseline which we use here is "Elastic infoGAN", 
In particular, we've adopted the implementation from https://github.com/utkarshojha/elastic-infogan
'''

import utils, torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class Attention_Random(nn.Module):
    """  
    Attnetion Layer with randomly selected mask ratio.
    Mask ratio is randomly sampled from uniform distribution(0.05, 0.2). 
    """
    def __init__(self):
        super(Attention_Random, self).__init__()
        self.mask_ratio = round(random.uniform(0.05, 0.2),2)
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
        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k) # (batch , 128, 128)
        p_attn = self.softmax(scores)
        if mask:
            p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value.transpose(-2,-1)) # 483*488 , 488*128
        
        return out.transpose(-2, -1), p_attn  # [Batch, 128, 483]


class Attention(nn.Module):
    """ 
    Attention layer with random mask. 
    Attention values randomly drop out depend on mask ratio. 
    """
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
        scores = torch.matmul(query.transpose(-2,-1), key) / math.sqrt(d_k) # (batch , 128, 128)
        p_attn = self.softmax(scores)
        if mask:
            p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value.transpose(-2,-1))
        return out.transpose(-2,-1), p_attn  # [Batch, 128, 483]



class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, len_discrete_code):
        super(Generator, self).__init__()
        self.input_dim = input_dim # z dim 
        self.output_dim = output_dim # 1 
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)

        self.fc1 = nn.Linear(self.input_dim + self.len_discrete_code, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128 * 488)
        self.bn2 = nn.BatchNorm1d(128 * 488)
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=8, stride=1, padding=1)
        self.bn_deconv = nn.BatchNorm1d(64)
        self.deconv2 = nn.ConvTranspose1d(64, self.output_dim, kernel_size=50, stride=6, padding=1)
        
        utils.initialize_weights(self)

    def forward(self, z, dist_code):
        x = torch.cat([z, dist_code], 1) 
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = x.view(-1, 128, 488) #reshape - [20, 128 ,488]
        x = F.relu(self.bn_deconv(self.deconv1(x)))
        x = torch.tanh(self.deconv2(x))
        return x 

class Front_end(nn.Module):
    def __init__(self, input_dim, segment):
        super(Front_end, self).__init__()
        self.input_dim = input_dim
        self.segment = segment
        
        if self.segment == 3:
            self.shape = 154
        if self.segment == 6:
            self.shape = 71
        if self.segment == 5:
            self.shape = 88
        if self.segment == 10:
            self.shape = 38    

        self.conv1 = nn.Conv1d(self.input_dim, 64, kernel_size=50, stride=6, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.utils.spectral_norm(nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=1))
        self.bn3 = nn.BatchNorm1d(128)

        self.fc_global = nn.Linear(128 * 488, 1024)
        self.fc_gbn = nn.BatchNorm1d(1024)

        self.fc_local = nn.Linear(128*self.shape, 1024)
        self.fc_lbn = nn.BatchNorm1d(1024)

        self.attn = Attention()
        
        utils.initialize_weights(self)
        
    def forward(self, input, mask, local): #[20, 1, 3000]
        x = F.leaky_relu(self.bn1(self.conv1(input)),0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)),0.2)
        x_2 = F.leaky_relu(self.bn3(self.conv3(x)),0.1)

        if local:
            x, attn_prob = self.attn(query=x, key=x_2, value=x_2, mask=mask) #[B, 128,71]
            x = x.reshape(-1, 128*self.shape)
            a = self.fc_local(x)
            a = F.leaky_relu(self.fc_lbn(a), 0.2)
        else:
            x, attn_prob = self.attn(query=x, key=x_2, value=x_2, mask=mask) # 이거 해보고 이상하면 conv1d_2빼고 해보기 
            out = x.reshape(-1, 128 * 488) 
            a = self.fc_global(out)
            a = F.leaky_relu(self.fc_gbn(a), 0.2)
        return a


class Discriminator(nn.Module):
    def __init__(self, output_dim=1):
        super(Discriminator, self).__init__()
        self.output_dim = output_dim
        
        self.fc = nn.Linear(1024, self.output_dim)

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        
        return x.view(-1,1) #[20,1]
 
class Latent_predictor(nn.Module):
    def __init__(self, len_discrete_code):
        super(Latent_predictor, self).__init__()
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)

        self.fc1 = nn.Linear(1024, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 =nn.Linear(128, self.len_discrete_code)

        utils.initialize_weights(self)

    def forward(self, input):
        a = F.leaky_relu(self.bn1(self.fc1(input)),0.2)
        b = self.fc2(a) #[20, 5]
        return a,b


## predictor : bottleneck structure 
class Latent_pred(nn.Module):
    def __init__(self):
        super(Latent_pred, self).__init__()

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 128)

        utils.initialize_weights(self)

    def forward(self, input):
        a = self.fc1(input)
        a = F.leaky_relu(a, 0.2) #[20, 128] 
        a = self.fc2(a) #[20, 128] 
        return a
