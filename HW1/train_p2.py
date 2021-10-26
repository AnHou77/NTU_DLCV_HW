import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import models
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchsummary import summary

import mean_iou_evaluate as miou

class p2(Dataset):
    def __init__(self, root):
        self.root = root

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        sat_filenames = glob.glob(os.path.join(root, f'*_sat.jpg'))
        mask_filenames = glob.glob(os.path.join(root, f'*_mask.png'))
        self.sat_filenames = sorted(sat_filenames)
        self.mask_filenames = sorted(mask_filenames)

        self.masks = miou.read_masks(root)
            
        if len(self.sat_filenames) == len(self.mask_filenames):
            self.len = len(self.sat_filenames)
                              
    def __getitem__(self, index):
        image_fn = self.sat_filenames[index]
        image = Image.open(image_fn)
            
        image = self.transform(image)
        label = self.masks[index]

        return image, label

    def __len__(self):
        return self.len

class VGG16_FCN32s(nn.Module):
    def __init__(self, num_class=7):
        super().__init__()

        self.features = torchvision.models.vgg16(pretrained=True).features

        self.fc = nn.Sequential(
            nn.Conv2d(512, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096,num_class,1)
        )
        self.upsample32x = nn.Upsample(scale_factor=32,mode='bilinear',align_corners=False)

        # for param in self.parameters():
        #     param.requires_grad = True
        
        # for param in self.features.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        fetures = self.features(x)
        fcn = self.fc(fetures)
        upsample = self.upsample32x(fcn)
        return upsample


def train(model, train_data, valid_data, epoch, save_path = './save_model/'):
     # Hyper parameter
    learning_rate = 1e-2
    weight_decay = 1e-4
    momentum = 0.9

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device used:', device)

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay= weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()

    best_acc = 0.0
    min_loss = np.inf

    for ep in range(1, epoch+1):
        print('Epoch {}/{}'.format(ep, epoch))
        print('-' * 10)

        # train
        train_loss = 0
        train_miou = 0
        first = True

        for batch_idx, (data, target) in enumerate(tqdm(train_data)):
            data, target = data.to(device), target.to(device,dtype=torch.long)
            optimizer.zero_grad()

            predict = model(data)
            predict = F.log_softmax(predict, dim=1)
        
            loss = criterion(predict, target)
            loss.backward()

            optimizer.step()

            label_pred = predict.max(dim=1)[1].data.cpu().numpy()
            target = target.cpu().numpy()
            # score = miou.mean_iou_score(label_pred,target,show_output=False)

            # Record Loss & Acc
            train_loss += loss.item()
            # train_miou += score
            if first:
                all_preds = label_pred
                all_labels = target
            else:
                all_preds = np.concatenate((all_preds,label_pred))
                all_labels = np.concatenate((all_labels,target))
            
            first = False
        
        train_loss = train_loss / len(train_data)
        train_miou = miou.mean_iou_score(all_preds,all_labels,False)
        # train_miou = train_miou / len(train_data)

        print(f"[ Train | {ep:03d}/{epoch:03d} ] loss = {train_loss:.5f}, mean_iou = {train_miou:.5f}")

        # Validation
        model.eval()

        valid_loss = 0.0
        valid_miou = 0.0

        first = True

        for (data, target) in tqdm(valid_data):
            data, target = data.to(device), target.to(device,dtype=torch.long)

            with torch.no_grad():
                predict = model(data)
                predict = F.log_softmax(predict, dim=1)

            loss = criterion(predict, target)
            
            label_pred = predict.max(dim=1)[1].data.cpu().numpy()
            target = target.cpu().numpy()
            # score = miou.mean_iou_score(label_pred,target,show_output=False)
            if first:
                all_preds = label_pred
                all_labels = target
            else:
                all_preds = np.concatenate((all_preds,label_pred))
                all_labels = np.concatenate((all_labels,target))
            
            first = False
            valid_loss += loss
            # valid_miou += score
        
        valid_loss = valid_loss / len(valid_data)
        # valid_miou = valid_miou / len(valid_data)
        valid_miou = miou.mean_iou_score(all_preds,all_labels,False)

        print(f"[ Valid | {ep:03d}/{epoch:03d} ] loss = {valid_loss:.5f}, mean_iou = {valid_miou:.5f}")
        # save the best model
        if valid_miou > best_acc:
            best_acc = valid_miou
            print('best acc: ',best_acc)
            # model_save = model.state_dict()
            # save_file = f"resnet152_{best_acc.item():.4f}.pth"
            # torch.save(model_save, os.path.join(save_path,save_file))
            # print('save model with acc:',best_acc)


def training():
    trainset = p2(root='data/p2_data/train')
    validset = p2(root='data/p2_data/validation')

    print('# images in trainset:', len(trainset))
    print('# images in validset:', len(validset))

    # Use the torch dataloader to iterate through the dataset
    trainset_loader = DataLoader(trainset, batch_size=20, shuffle=True, num_workers=4)
    validset_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)

    # get some random training images
    dataiter = iter(trainset_loader)
    images, labels = dataiter.next()

    print('(Trainset) Image tensor in each batch:', images.shape, images.dtype)
    print('(Trainset) Label tensor in each batch:', labels.shape, labels.dtype)

    model = VGG16_FCN32s(7).cuda()

    train(model, trainset_loader, validset_loader, 50)
    # summary(model,(3,512,512))



if __name__ == '__main__':
    training()