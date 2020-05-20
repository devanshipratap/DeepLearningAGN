#2020-5-20 by Joshua Yao-Yu Lin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from  torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from tensorboardX import SummaryWriter
from uncertainties import ufloat

### file paths
folder = './new_Full_train/'
test_folder = 'test/'
train_folder = 'train/'
test_src_folder = 'data_test_src'
LC_path = './QSO_LCs/QSO_S82/'
files = os.listdir(LC_path)
loaded_model_path = './saved_model/2020-01-28new_redshift_resnet18.mdl'



### choose cpu as device
device = torch.device('cpu')


### load saved model
if os.path.exists(loaded_model_path):
    net = torch.load(loaded_model_path, map_location=device)
    print('loaded mdl！')
else:
    print('No model to load. Should stop!')


### batch_size
glo_batch_size = 1
test_num_batch = 1

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([
            transforms.ToTensor(), # scale to [0,1] and convert to tensor
            normalize,
            ])
target_transform = torch.Tensor

### datasets
class BHDataset(Dataset): # torch.utils.data.Dataset
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_folder = 'train'#'data_train'
        self.test_folder = 'test'#'data_test'
        #self.df = pd.read_csv(self.root_dir + '/clean_full_data.csv')

        if self.train:
            self.path = os.path.join(self.root_dir, self.train_folder)
            self.df = pd.read_csv(self.path + '/train.csv')
            #self.length = TRAINING_SAMPLES
        else:
            self.path = os.path.join(self.root_dir, self.test_folder)
            self.df = pd.read_csv(self.path + '/test.csv')
            #self.length = TESTING_SAMPLES

    def __getitem__(self, index):

        #print(self.df['ID'])
        ID = self.df['ID'].iloc[[index]]
        M = self.df['M'].iloc[[index]]
        z = self.df['z'].iloc[[index]]
        #print("ID:", ID.values[0])
        img_path = self.path + '/LC_images_' + str(ID.values[0]) + '.npy'
        img = np.load(img_path)
        image = np.zeros((3, 224, 224))
        for i in range(3):
            image[i, :, :] += img
        return image, M.values, ID.values[0], z.values

    def __len__(self):
        return self.df.shape[0]
        #return self.length




### test loader set up
test_loader = torch.utils.data.DataLoader(BHDataset(folder, train=False, transform=data_transform, target_transform=target_transform),
                    batch_size = glo_batch_size, shuffle = True
                    )


### create lists to store prediction
target_list = []
output_list = []
error_list = []
ID_list = []


### eval mode for net
net.eval()

for batch_idx, (data, BH_Mass, ID, z) in enumerate(test_loader):
    data, target = data.float(), z.float()
    #data, target = Variable(data).cuda(), Variable(target).cuda()
    LC_ID = str(ID.numpy()[0])

    output = net(data)
    cpu_output = output.data.cpu().numpy()[0][0]
    cpu_uncertainity = np.exp(output.data.cpu().numpy()[0][1])**0.5
    cpu_target = target.data.cpu().numpy()[0][0]

    target_list.append(cpu_target)
    output_list.append(cpu_output)
    error_list.append(cpu_uncertainity)
    ID_list.append(LC_ID)
    del output
    del data
    del target
    del ID
    gc.collect()


df = pd.DataFrame()
df['ID'] = ID_list
df['z_ground_truth'] = target_list
df['z_prediction'] = output_list
df['z_error_prediction'] = error_list


save_path = './Prediction_table/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

df.to_csv(save_path + 'v0520_z_nll.csv')
