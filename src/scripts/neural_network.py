"""
Script for training a neural network (resnet18) using pytorch
"""

import os
import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lenstronomy.Util.image_util as image_util
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms

# Specify the data folder and hyper-parameters for training
FOLDER = "./full_train/"
EPOCH = 60
GLO_BATCH_SIZE = 10
TEST_NUM_BATCH = 50

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DATA_TRANSFORM = transforms.Compose([
    transforms.ToTensor(), # scale to [0,1] and convert to tensor
    NORMALIZE, ])
TARGET_TRANSFORM = torch.Tensor


# Using pytorch dataloader to read the data for training and testing
class BHDataset(Dataset):
    """
    Class to model training or testing data sets
    """
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
        """
        Function to specify the path, and lookup tables (csv) for the dataset
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_folder = 'train/'#'data_train'
        self.test_folder = 'test/'#'data_test'
        #self.df = pd.read_csv(self.root_dir + '/clean_full_data.csv')

        if self.train:
            self.path = os.path.join(self.root_dir, self.train_folder)
            self.df = pd.read_csv(self.path + 'train.csv')
            self.df['M'] = self.df.M.mask(self.df.M == 0, 8.8)
            #self.length = TRAINING_SAMPLES
        else:
            self.path = os.path.join(self.root_dir, self.test_folder)
            self.df = pd.read_csv(self.path + 'test.csv')
            self.df['M'] = self.df.M.mask(self.df.M == 0, 8.8)
            #self.length = TESTING_SAMPLES

    def __getitem__(self, index):
        """
        Function to obtain the data, and the labels in the given path
        """
        ID = self.df['ID'].iloc[[index]]
        M = self.df['M'].iloc[[index]]
        try:
            img_path = self.path + 'LC_images_' + str(ID.values[0]) + '.npy'
            img = np.load(img_path)

        except:
            img = np.zeros((224, 224))
            print("missing", str(ID.values[0]))
        image = np.zeros((3, 224, 224))
        for i in range(3):
            image[i, :, :] += img
        return image, M.values

    def __len__(self):
        """
        Function to get the length of the dataframe
        """
        return self.df.shape[0]

# Train loader loads the data only from training set
TRAIN_LOADER = torch.utils.data.DataLoader(BHDataset(FOLDER, train=True,
                                                     transform=DATA_TRANSFORM,
                                                     target_transform=TARGET_TRANSFORM),
                                           batch_size=GLO_BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
    # Specify the number of output parameters (AGN mass only for example, so there is only one here)
    DSET_CLASSES_NUMBER = 1

    # Using a pre-trained resnet18
    NET = models.resnet18(pretrained=True)

    # Modify the last layer of the network to output the parameters we wish to train
    NUM_FTRS = NET.fc.in_features
    NET.fc = nn.Linear(in_features=NUM_FTRS, out_features=DSET_CLASSES_NUMBER)

    # Loss function for training
    LOSS_FN = nn.MESLoss(reduction='elementwise_mean')

    # Putting the neural network on GPU
    NET.cuda()

    # Specify the optimizer and the initial learning rate
    OPTIMIZER = optim.Adam(NET.parameters(), lr=2*1e-4)
    TB = SummaryWriter()

    # Set best_accuracy first, and overwrite it during training
    BEST_ACCURACY = float("inf")

    # Looping through the training epoch.
    # During each epoch, the neural network goes through the whole training set
    for epoch in range(EPOCH):

        # Set the network to training phase
        NET.train()
        total_loss = 0.0
        total_counter = 0
        total_rms = 0

        for batch_idx, (data, target) in enumerate(tqdm(TRAIN_LOADER, total=len(TRAIN_LOADER))):
            data, target = data.float(), target.float()
            data, target = Variable(data).cuda(), Variable(target).cuda()

            OPTIMIZER.zero_grad()
            output = NET(data)
            loss = LOSS_FN(output, target)

            square_diff = (output - target) #((output - target)**2)**(0.5)
            total_rms += square_diff.std(dim=0)
            total_loss += loss.item()
            total_counter += 1

            loss.backward()
            OPTIMIZER.step()

        # Collect RMS over each label
        avg_rms = total_rms / (total_counter)
        avg_rms = avg_rms.cpu()
        avg_rms = (avg_rms.data).numpy()
        for i in range(len(avg_rms)):
            TB.add_scalar('rms %d' % (i+1), avg_rms[i])

        # Print test loss and test RMS
        print(epoch, 'Train loss (average per batch wise):', total_loss/(total_counter),
              ' RMS (average per batch wise):', np.array_str(avg_rms, precision=3))

        with torch.no_grad():

            # Set the network to evaluating phase
            NET.eval()
            total_loss = 0.0
            total_counter = 0
            total_rms = 0

            # Test loader load the data only from test set
            test_loader = torch.utils.data.DataLoader(BHDataset(FOLDER, train=False,
                                                                transform=DATA_TRANSFORM,
                                                                target_transform=TARGET_TRANSFORM),
                                                      batch_size=GLO_BATCH_SIZE, shuffle=True)

            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.float(), target.float()
                data, target = Variable(data).cuda(), Variable(target).cuda()

                # Getting prediction from neural network
                pred = NET(data)
                loss = LOSS_FN(pred, target)
                square_diff = (pred - target)
                total_rms += square_diff.std(dim=0)
                total_loss += loss.item()
                total_counter += 1

                if batch_idx % TEST_NUM_BATCH == 0 and batch_idx != 0:
                    TB.add_scalar('test_loss', loss.item())
                    break

            # Collect RMS over each label
            avg_rms = total_rms / (total_counter)
            avg_rms = avg_rms.cpu()
            avg_rms = (avg_rms.data).numpy()
            for i in range(len(avg_rms)):
                TB.add_scalar('rms %d' % (i+1), avg_rms[i])

            # print test loss and test RMS
            print(epoch, 'Test loss (average per batch wise):', total_loss/(total_counter),
                  'RMS (average per batch wise):', np.array_str(avg_rms, precision=3))

            # Save the best fit models
            if total_loss/(total_counter) < BEST_ACCURACY:
                BEST_ACCURACY = total_loss/(total_counter)
                datetime_today = str(datetime.date.today())
                torch.save(NET, './saved_model/' + datetime_today +'resnet18.mdl')
                print("saved to " + datetime_today + "resnet18.mdl" + " file.")

    TB.close()
