# 2020-05-26 at Home
from efficientnet_pytorch import EfficientNet
# from albumentations.pytorch import ToTensor
# from albumentations import (
#     Compose, HorizontalFlip, CLAHE, HueSaturationValue,
#     RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize,
#     ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
#     RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
#     IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip
# )
import os
import torch
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from glob import glob
import torchvision
from torch.utils.data import Dataset
from  torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import time
#from tqdm.notebook import tqdm
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime
from sklearn import metrics
#import cv3
import gc
import torch.nn.functional as F


### seed for reproducibility

seed = 42
print(f'setting everything to seed {seed}')
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


### EfficientNet


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1')
        # 1280 is the number of neurons in last layer. is diff for diff. architecture
        #self.fc1 = nn.Linear(1280 + 1, 1280)
        self.dense_output = nn.Linear(1280, num_classes)




    def forward(self, x):
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        # feat = torch.cat((feat, z), dim=1)
        # feat = F.relu(self.fc1(feat))

        return self.dense_output(feat)


class M_Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1')
        # 1280 is the number of neurons in last layer. is diff for diff. architecture
        self.fc1 = nn.Linear(1280 + 1, 1280)
        self.dense_output = nn.Linear(1280, num_classes)




    def forward(self, x, z):
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        feat = torch.cat((feat, z), dim=1)
        feat = F.relu(self.fc1(feat))

        return self.dense_output(feat)




folder = "/media/joshua/Milano/Efficient_Full_train/"
EPOCH = 200
glo_batch_size = 8
test_num_batch = 50


loaded_model_path = '/home/joshua/Documents/git_work_zone/Black_holes_NN/saved_model/2020-05-30nll_zpred_ENetb1-v1.mdl'
if os.path.exists(loaded_model_path):
   z_net = torch.load(loaded_model_path)
   print('loaded mdl！')


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([
            transforms.ToTensor(), # scale to [0,1] and convert to tensor
            normalize,
            ])
target_transform = torch.Tensor

class BHDataset(Dataset): # torch.utils.data.Dataset
    def __init__(self, root_dir, version='v1', train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        if version == None or version == 'v1':
            self.train_folder = 'train'#'data_train'
            self.test_folder = 'test'#'data_test'
        elif version == 'v2':
            self.train_folder = 'train_v2'#'data_train'
            self.test_folder = 'test_v2'#'data_test'
        elif version == 'v3':
            self.train_folder = 'train_v3'#'data_train'
            self.test_folder = 'test_v3'#'data_test'

        #self.df = pd.read_csv(self.root_dir + '/clean_full_data.csv')

        if self.train:
            self.path = os.path.join(self.root_dir, self.train_folder)
            self.df = pd.read_csv(self.path + '/train_err.csv')
            self.df['M'] = self.df.M.mask(self.df.M == 0,8.8)
            #self.length = TRAINING_SAMPLES
        else:
            self.path = os.path.join(self.root_dir, self.test_folder)
            self.df = pd.read_csv(self.path + '/test_err.csv')
            self.df['M'] = self.df.M.mask(self.df.M == 0,8.8)
            #self.length = TESTING_SAMPLES

    def __getitem__(self, index):

        #print(self.df['ID'])
        ID = self.df['ID'].iloc[[index]]
        M = self.df['M'].iloc[[index]]
        M_err = self.df['M_ERR'].iloc[[index]]
        z = self.df['z'].iloc[[index]]
        #print("ID:", ID.values[0])
        #print("redshift:", z)
        img_path = self.path + '/LC_images_' + str(ID.values[0]) + '.npy'
        img = np.load(img_path)
        image = np.zeros((3, 224, 224))
        for i in range(3):
            image[i, :, :] += img
        return image, M.values, M_err.values, z.values

    def __len__(self):
        return self.df.shape[0]
        #return self.length


train_loader = torch.utils.data.DataLoader(BHDataset(folder, version='v1', train=True, transform=data_transform, target_transform=target_transform),
                    batch_size = glo_batch_size, shuffle = True
                    )


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def nll_diagonal(target, mu, logvar):
    """Evaluate the NLL for single Gaussian with diagonal covariance matrix
    Parameters
    ----------
    target : torch.Tensor of shape [batch_size, Y_dim]
        Y labels
    mu : torch.Tensor of shape [batch_size, Y_dim]
        network prediction of the mu (mean parameter) of the BNN posterior
    logvar : torch.Tensor of shape [batch_size, Y_dim]
        network prediction of the log of the diagonal elements of the covariance matrix
    Returns
    -------
    torch.Tensor of shape
        NLL values
    """
    precision = torch.exp(-logvar)
    # Loss kernel
    loss = precision * (target - mu)**2.0 + logvar
    # Restore prefactors
    loss += np.log(2.0*np.pi)
    loss *= 0.5
    return torch.mean(torch.sum(loss, dim=1), dim=0)

if __name__ == '__main__':

    dset_classes_number = 2
    device = 'cuda'
    net = M_Net(num_classes=dset_classes_number).to(device)

    #loaded_model_path = './saved_model/redshift_resnet18.mdl'
	#save_model_path = './saved_model/'
    #if not os.path.exists(save_model_path):
	#os.mkdir(save_model_path)
    #if os.path.exists(loaded_model_path):
    #    net = torch.load(loaded_model_path)
    #    print('loaded mdl！')
    loss_fn = nn.MSELoss(reduce=True, size_average=True)
    #loss_fn = weighted_mse_loss()
    net.cuda()

    optimizer = optim.AdamW(net.parameters(), lr = 1e-4)
    tb = SummaryWriter()

    best_accuracy = float("inf")


    #if os.path.exists('./saved_model/resnet18.mdl'):
        #net = torch.load('./saved_model/resnet18.mdl')
        #print('loaded mdl!')

    for epoch in range(EPOCH):

        net.train()
        total_loss = 0.0
        total_counter = 0
        total_rms = 0

        for batch_idx, (data, BH_Mass, M_err, z) in enumerate(tqdm(train_loader, total = len(train_loader))):
            data, BH_Mass, M_err, z = data.float(), BH_Mass.float(), M_err.float(), z.float()
            data, BH_Mass, M_err, z = Variable(data).cuda(), Variable(BH_Mass).cuda(), Variable(M_err).cuda(), Variable(z).cuda()

            optimizer.zero_grad()
            z_pred = z_net(data)
            #print("z_pred", z_pred, z_pred[:, 0])

            #print("shapeinfo", data.size(), z_pred[:, 0].size(), z.size())
            output = net(data, z_pred[:, 0].unsqueeze(1))

            #loss = weighted_mse_loss(output, BH_Mass, weight = M_err**-2)#loss_fn(output, BH_Mass)
            loss = nll_diagonal(target=BH_Mass, mu=output[:, 0].unsqueeze(1), logvar= output[:, 1].unsqueeze(1))
            square_diff = (output - BH_Mass) #((output - BH_Mass)**2)**(0.5)
            total_rms += square_diff.std(dim=0)
            total_loss += loss.item()
            total_counter += 1

            loss.backward()
            optimizer.step()

        # Collect RMS over each label
        avg_rms = total_rms / (total_counter)
        avg_rms = avg_rms.cpu()
        avg_rms = (avg_rms.data).numpy()
        for i in range(len(avg_rms)):
            tb.add_scalar('rms %d' % (i+1), avg_rms[i])

        # print test loss and tets rms
        print(epoch, 'Train loss (averge per batch wise):', total_loss/(total_counter), ' RMS (average per batch wise):', np.array_str(avg_rms, precision=3))

        with torch.no_grad():
            net.eval()
            total_loss = 0.0
            total_counter = 0
            total_rms = 0

            test_loader = torch.utils.data.DataLoader(BHDataset(folder, version='v1', train=False, transform=data_transform, target_transform=target_transform),
                        batch_size = glo_batch_size, shuffle = True
                        )

            for batch_idx, (data, BH_Mass, M_err, z) in enumerate(test_loader):
                data, BH_Mass, M_err, z = data.float(), BH_Mass.float(), M_err.float(), z.float()
                data, BH_Mass, M_err, z = Variable(data).cuda(), Variable(BH_Mass).cuda(), Variable(M_err).cuda(), Variable(z).cuda()

                #pred [batch, out_caps_num, out_caps_size, 1]
                z_pred = z_net(data)
                pred = net(data, z_pred[:, 0].unsqueeze(1))
                #pred = net(data, z)
                #loss = weighted_mse_loss(pred, BH_Mass, weight = M_err**-2)#loss_fn(output, BH_Mass)
                loss = nll_diagonal(target=BH_Mass, mu=pred[:, 0].unsqueeze(1), logvar= pred[:, 1].unsqueeze(1))
                square_diff = (pred - BH_Mass)
                total_rms += square_diff.std(dim=0)
                total_loss += loss.item()
                total_counter += 1

                if batch_idx % test_num_batch == 0 and batch_idx != 0:
                    tb.add_scalar('test_loss', loss.item())
                    break

            # Collect RMS over each label
            avg_rms = total_rms / (total_counter)
            avg_rms = avg_rms.cpu()
            avg_rms = (avg_rms.data).numpy()
            for i in range(len(avg_rms)):
                tb.add_scalar('rms %d' % (i+1), avg_rms[i])

            # print test loss and tets rms
            print(epoch, 'Test loss (averge per batch wise):', total_loss/(total_counter), ' RMS (average per batch wise):', np.array_str(avg_rms, precision=3))
            if total_loss/(total_counter) < best_accuracy:
                best_accuracy = total_loss/(total_counter)
                datetime_today = str(datetime.date.today())
                torch.save(net, './saved_model/' + datetime_today + 'mass_from_z_ENetb1-v1.mdl')
                print("saved to " + "mass_from_z_ENetb1-v1.mdl" + " file.")

    tb.close()
