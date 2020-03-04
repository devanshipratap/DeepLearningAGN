import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from  torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
import lenstronomy.Util.image_util as image_util
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import torch.utils.model_zoo as model_zoo
from tensorboardX import SummaryWriter

folder = "./v0812_redshift_aug_Full_train/"
EPOCH = 60
glo_batch_size = 10
test_num_batch = 50

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([
            transforms.ToTensor(), # scale to [0,1] and convert to tensor
            normalize,
            ])
target_transform = torch.Tensor

class BHDataset(Dataset): # torch.utils.data.Dataset
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
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
            self.df['M'] = self.df.M.mask(self.df.M == 0,8.8)
            #self.length = TRAINING_SAMPLES
        else:
            self.path = os.path.join(self.root_dir, self.test_folder)
            self.df = pd.read_csv(self.path + 'test.csv')
            self.df['M'] = self.df.M.mask(self.df.M == 0,8.8)
            #self.length = TESTING_SAMPLES

    def __getitem__(self, index):

        #print(self.df['ID'])
        ID = self.df['ID'].iloc[[index]]
        M = self.df['M'].iloc[[index]]
        #print("ID:", ID.values[0])
        try:
            img_path = self.path + 'LC_images_' + str(ID.values[0]) + '.npy'
            img = np.load(img_path)
            # sigma_to_noise_ratio = 100
            # total_flux = sum(sum(img))
            # N_pix = img.size
            # sigma_n = total_flux / (np.sqrt(N_pix) * sigma_to_noise_ratio)
            # gaussian_noise = sigma_n * np.random.randn(img.shape[0], img.shape[1])
            # img += gaussian_noise
            # plt.imshow(img)
            # plt.show()

        except:
            img = np.zeros((224, 224))
            print("missing", str(ID.values[0]))
        image = np.zeros((3, 224, 224))
        for i in range(3):
            image[i, :, :] += img
        return image, M.values

    def __len__(self):
        return self.df.shape[0]
        #return self.length


train_loader = torch.utils.data.DataLoader(BHDataset(folder, train=True, transform=data_transform, target_transform=target_transform),
                    batch_size = glo_batch_size, shuffle = True
                    )

if __name__ == '__main__':

    dset_classes_number = 1
    net = models.resnet34(pretrained=True)
    #net = model_zoo.xception(pretrained= True)
    num_ftrs = net.fc.in_features
    net.fc= nn.Linear(in_features=num_ftrs, out_features=dset_classes_number)
    #loaded_model_path = './saved_model/redshift_aug_resnet18.mdl'


    # if os.path.exists(loaded_model_path):
    #     net = torch.load(loaded_model_path)
    #     print('loaded mdl！')
    loss_fn = nn.L1Loss(reduction='elementwise_mean')

    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr = 2*1e-4)
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

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, total = len(train_loader))):
            data, target = data.float(), target.float()
            data, target = Variable(data).cuda(), Variable(target).cuda()

            optimizer.zero_grad()
            output = net(data)
            loss = loss_fn(output, target)

            square_diff = (output - target) #((output - target)**2)**(0.5)
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

            test_loader = torch.utils.data.DataLoader(BHDataset(folder, train=False, transform=data_transform, target_transform=target_transform),
                        batch_size = glo_batch_size, shuffle = True
                        )

            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.float(), target.float()
                data, target = Variable(data).cuda(), Variable(target).cuda()

                #pred [batch, out_caps_num, out_caps_size, 1]
                pred = net(data)
                loss = loss_fn(pred, target)
                square_diff = (pred - target)
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
                torch.save(net, './saved_model/' + datetime_today +'mae_k_corr_redshift_aug_resnet34.mdl')
                print("saved to " + datetime_today + "mae_k_corr_redshift_aug_resnet34.mdl" + " file.")

    tb.close()
