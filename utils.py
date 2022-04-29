import torch
import torch.nn as nn

import numpy as np
import random
from tsaug import Quantize


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


# make data ranage between [-1~1]
def normalization(data):
    d = 2.*(data - np.min(data))/np.ptp(data)-1
    return d

def standardization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    epsilon = 0.000001

    data = (data - mean) / (std + epsilon)

    return data

def count_data(data):
    full_list = []
    for _ , (_, label) in enumerate(data):
        full_list = np.append(full_list, label)
        (unique, count) = np.unique(full_list, return_counts=True)
    return unique, count

def make_weights(dataset):
    (_, count) = count_data(dataset)
    class_weights = 1/count
    print(class_weights)
    sample_weights = [0]*len(dataset)

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    (uni, cou) = count_data(dataset)
    print(uni, cou)
    sample_weights = torch.DoubleTensor(sample_weights)
    return sample_weights


##############################################################
# augmentation method for data_loader 
##############################################################
def addNoise(data, noise_scale_rate=1.0):
            return data + np.random.normal(loc = 0, scale = noise_scale_rate, size = data.shape)
               
def quantize(data, n_levels):
    my_augmenter = (Quantize(n_levels =[n_levels]))
    x = np.reshape(data,(1,3000))
    aug = my_augmenter.augment(x)
    aug = np.reshape(aug, (3000,1))
    return aug

def magScale(data, mag_scale_rate=0.7):
            return data * np.random.normal(loc = 0.1, scale = mag_scale_rate, size = data.shape)
    

def timeShift(data, shift_ratio):
    '''
    Args:
        - data : EEG + EOG concated data  # [3000, 2]
        - shift_ratio : shifting ratio for shifting range 

    The timeshift augmentation process is: 
    1. Within len(data)*shift_ratio len, 
    2. for {-shfit range , +shift range}, randomly selet shift ranage. 
    3. fill one side (front/back) of signal with zero.
    Note that since we select shift range randomly, we don't know which side(front or back) would be filled with zero value.
    '''

    # for (-shfit range , + shift range), randomly selet shift ranage. 
    shift_range = random.randint(-len(data) * shift_ratio, len(data) * shift_ratio)
    
    # a is value for fill zero at front of signal , b is value for fill zero at back of signal 
    a = -min(0, shift_range) # front에 0 채움 
    b = max(0, shift_range) # back에 0 채움  
    data_tmp = torch.from_numpy(np.pad(data[:][0], (a, b), "reflect")) #  EEG tmp  [3000, 1]
    data_tmp_2 = torch.from_numpy(np.pad(data[:][1], (a, b), "reflect")) #  EEG tmp  [3000, 1]
    
    data[:][0] = data_tmp[:len(data_tmp) - a] if a else data_tmp[b:] # EEG
    data[:][1] = data_tmp[:len(data_tmp_2) - a] if a else data_tmp_2[b:] # EOG
    del data_tmp
    del data_tmp_2

    return data 
        

def get_noisy_signal(data, add_noise=True, noise_scale_rate=1.0, 
                    mag_scale=False, mag_scale_rate=0.7,
                    time_shift=False, shift_ratio=0.2,
                    quantize = False):
    '''
    Generate noisy signal for positive pair. 

    Args:
        - data[nparray] : signal data  
        - add_noise[bool]: Wether to use additional nosie augmentation. 
        - noise_scale_rate[float] : The size of noise value.    
            
        - mag_sacle[bool]: Wether to use magnitude scaling agumentation 
        - mag_scale_rate[float] : The size of magnitude scaling.

        - shift[bool]: Wheter to use shfting agumentation 
        - shift_ratio[float] : The size of shfting.
        
    '''

    if add_noise:
        data  = addNoise(data, noise_scale_rate)

    if mag_scale: 
        data = magScale(data, mag_scale_rate) 

    if time_shift: 
        data = timeShift(data, shift_ratio)

    if quantize:
        data = quantize(data, n_levels=10)
    
    return data