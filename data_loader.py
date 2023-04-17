from numpy.core.fromnumeric import transpose

from torch.utils.data import DataLoader
from torch.utils.data import Dataset 

import os
import numpy as np
import glob

import utils  
from utils import get_noisy_signal, normalization

########## Data Loader for EDF signal Data ##########
'''
DataLoader for the raw signal EDF format data.
Positive pair generation using general augmentation techniques are included (e.g. add noise, scaling, quantize, etc.).
Adjust the parameter in the SleepEDFx class to generate positive pairs with specific augmentations. Combined use of various augmentations is also available.
'''

class SleepEDFX(Dataset):
    def __init__(self, root_dir, sensor_dir_1, sensor_dir_2,  
                get_aug_signal,
                add_noise, noise_scale_rate, 
                mag_scale, mag_scale_rate, 
                shift=False, shift_ratio=0.5,
                quantize = False, 
                normalization=False,
                ):
        
        '''
        Dataset Class for loading SleepEDFX data. 
        It can be applied to other sleep datasets consisted of 1 epoch (30 sec) sleep signal.
        (e.g. ISRUC, HMC, ... )
     
        Arges:
            root_dir : directory of data 
            sensor_dir_1 : directory of sensor 1 
            sensor_dir_2 : directory of sensor 1 
            
            get_aug_signal[bool] : Whether to generate positive pairs using general augmentation method. 
            add_noise[bool]: Whether to use 'add noise' augmentation. 
            noise_scale_rate[float] : The size of noise value. 
            the magnitude of noise value.   
            
            mag_sacle[bool]: Whether to use 'magnitude scaling' agumentation 
            mag_scale_rate[float] : The size of magnitude scaling.
            shift[bool]: Wheter to use 'shift' agumentation 
            shift_ratio[float] :  The size of shfting.
            quantize[bool]: Wheter to use 'quantize' agumentation 
            normalization[bool]: Wheter to apply normalization on the input data.
        '''
        
        # data path  
        self.root_dir = root_dir 
        self.sensor_dir_1 = sensor_dir_1     
        self.sensor_dir_2 = sensor_dir_2     
        
        sensor_path_1 = os.path.join(root_dir, sensor_dir_1)
        self.file_names_1 = glob.glob(os.path.join(sensor_path_1,'*.npz'))
        
       #  sensor_2 path 
        if self.sensor_dir_2 != None:
            sensor_path_2 = os.path.join(root_dir, sensor_dir_2)
            self.file_names_2 = glob.glob(os.path.join(sensor_path_2,'*.npz'))
    
        # augmentation setting 
        self.add_noise = add_noise
        self.mag_scale = mag_scale
        self.noise_scale_rate = noise_scale_rate 
        self.mag_scale_rate = mag_scale_rate 
        self.shift = shift
        self.shift_ratio = shift_ratio
        self.quantize = quantize 
        self.get_aug_signal = get_aug_signal

        # normalization 
        self.normalization = normalization   

    def __len__(self):
        return len(self.file_names_1)
        
        
    def __getitem__(self, idx):
        
        file = np.load(self.file_names_1[idx]) # signal 1 (EEG here)
        file_2 = np.load(self.file_names_2[idx]) # signal 2 (EOG here)
        x_org = file['x']  
        x_org_2 = file_2['x']
        target = file['y']

        x_aug = x_org # init augmented data 1 
        x_aug_sec = x_org_2  # init augmented data 2 

        if self.get_aug_signal == True: # generate augmented signal using common augmentation method 
            # data augmentation for signal 1 
            x_aug = get_noisy_signal(file['x'], 
                                 add_noise = self.add_noise, noise_scale_rate = self.noise_scale_rate,
                                 mag_scale = self.mag_scale, mag_scale_rate = self.mag_scale_rate)
                                 
            x_aug_1 = get_noisy_signal(file['x'], 
                                 add_noise = self.add_noise, noise_scale_rate = self.noise_scale_rate,
                                 mag_scale = self.mag_scale, mag_scale_rate = self.mag_scale_rate)

            if self.normalization:
                x_aug = normalization(x_aug)
                x_aug_1 = normalization(x_aug_1)

            # data augmentation for signal 2 
            x_aug_2 = get_noisy_signal(file_2['x'], 
                                add_noise = self.add_noise, noise_scale_rate = self.noise_scale_rate,
                                mag_scale = self.mag_scale, mag_scale_rate = self.mag_scale_rate,
                                quantize=self.quantize)
            x_aug_2_2 = get_noisy_signal(file_2['x'], 
                                add_noise = self.add_noise, noise_scale_rate = self.noise_scale_rate,
                                mag_scale = self.mag_scale, mag_scale_rate = self.mag_scale_rate,
                                quantize=self.quantize)
        
            if self.normalization:
                x_aug = normalization(x_aug_2)
                x_aug_2_2 = normalization(x_aug_2_2)
            
            x_aug = np.concatenate((x_aug, x_aug_2), axis=1) # [3000, 2]
            x_aug_sec = np.concatenate((x_aug_1, x_aug_2_2), axis=1) # [3000, 2]
        
        if self.normalization:
            x_org = normalization(x_org)
            x_org_2 = normalization(x_org_2)
            
        x_org = np.concatenate((x_org, x_org_2), axis=1) # [3000, 2]

        if self.shift:
            x_aug_tmp = x_org # raw signal 
            x_aug = utils.timeShift(x_aug_tmp, shift_ratio = self.shift_ratio) 
            x_aug_sec = utils.timeShift(x_aug_tmp, shift_ratio = self.shift_ratio)
            
        return x_org.transpose(1,0), x_aug.transpose(1,0), x_aug_sec.transpose(1,0), target # transpose data 


def dataloader(root_dir, sensor_dir, sensor_dir_2, batch_size, normalization):

    '''
    Args:
        - root_dir : root dir
        - sensor_dir : sensor dir 1 (EEG here)
        - sensor_dir_2 : sensor dir 2 (EOG here)
        - batch_size : size of batch 
    '''

    SleepEDFX_dataset = SleepEDFX(root_dir = root_dir, sensor_dir_1 = sensor_dir, sensor_dir_2 = sensor_dir_2, 
                                get_aug_signal = False,
                                add_noise = False, noise_scale_rate = 1.0, 
                                mag_scale = False, mag_scale_rate = 0.5,
                                shift=False, shift_ratio=0.5,
                                quantize = False,
                                normalization = normalization)
    
    data_loader = DataLoader(dataset = SleepEDFX_dataset, 
                            batch_size = batch_size,
                            shuffle = True 
                            )

    
    return data_loader    
    
    
    
