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
from torchsummary import summary
import skimage.io as imgio
import warnings

# deal with image output warning
warnings.filterwarnings('ignore')

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

        self.img_indices = []

        for fn in self.sat_filenames:
            fn = fn.replace(root,"")
            self.img_indices.append(fn.replace("/","")[:-8])

        self.masks = miou.read_masks(root)
            
        if len(self.sat_filenames) == len(self.mask_filenames):
            self.len = len(self.sat_filenames)
                              
    def __getitem__(self, index):
        image_fn = self.sat_filenames[index]
        image = Image.open(image_fn)
            
        image = self.transform(image)
        label = self.masks[index]

        img_index = self.img_indices[index]

        return image, label, img_index

    def __len__(self):
        return self.len

class VGG16_FCN32s(nn.Module):
    def __init__(self, num_class=7):
        super().__init__()

        self.features = torchvision.models.vgg16(pretrained=True).features

        self.fc = nn.Sequential(
            nn.Conv2d(512, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096,num_class, 1)
        )
        self.upsample32 = nn.Upsample(scale_factor=32,mode='bilinear',align_corners=False)
        # self.upsample32 = nn.ConvTranspose2d(num_class, num_class, 32 , 32)

    def forward(self, x):
        fetures = self.features(x)
        fcn = self.fc(fetures)
        upsample = self.upsample32(fcn)
        return upsample

class VGG16_FCN8s(nn.Module):
    def __init__(self, num_class=7):
        super().__init__()

        # self.features = torchvision.models.vgg16(pretrained=True).features.children()

        # # self.pool3 = nn.Sequential(*list(self.features)[:17])

        # # self.pool4 = nn.Sequential(*list(self.features)[17:24])

        # # self.pool5 = nn.Sequential(*list(self.features)[24:])

        self.model = torchvision.models.vgg16(pretrained=True)
        self.pool3 = nn.Sequential(*list(self.model.features.children())[:17])
        self.pool4 = nn.Sequential(*list(self.model.features.children())[17:24])
        self.pool5 = nn.Sequential(*list(self.model.features.children())[24:])

        self.fc = nn.Sequential(
            nn.Conv2d(512, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096,num_class, 1)
        )

        self.pool3_conv = nn.Conv2d(256, num_class, 1)
        self.pool4_conv = nn.Conv2d(512, num_class, 1)

        self.upsample2x = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.upsample4x = nn.Upsample(scale_factor=4,mode='bilinear',align_corners=False)
        self.upsample8x = nn.Upsample(scale_factor=8,mode='bilinear',align_corners=False)
        # self.upsample32 = nn.ConvTranspose2d(num_class, num_class, 32 , 32)

    def forward(self, x):
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool4_2x = self.upsample2x(pool4)
        pool5 = self.pool5(pool4)
        conv7 = self.fc(pool5)
        conv7_4x = self.upsample4x(conv7)

        pool3 = self.pool3_conv(pool3)
        pool4_2x = self.pool4_conv(pool4_2x)
        upsample = self.upsample8x(pool3+pool4_2x+conv7_4x)
        return upsample

def mask_lable_to_rgb(labels):
    rgbs = np.empty((len(labels), 512, 512, 3))
    
    for i, p in enumerate(labels):
        rgbs[i, p == 0] = [0,255,255]   # (Cyan: 011) Urban land 
        rgbs[i, p == 1] = [255,255,0]   # (Yellow: 110) Agriculture land 
        rgbs[i, p == 2] = [255,0,255]   # (Purple: 101) Rangeland 
        rgbs[i, p == 3] = [0,255,0]     # (Green: 010) Forest land
        rgbs[i, p == 4] = [0,0,255]     # (Blue: 001) Water
        rgbs[i, p == 5] = [255,255,255] # (White: 111) Barren land
        rgbs[i, p == 6] = [0,0,0]       # (Black: 000) Unknown 
    
    rgbs = rgbs.astype(np.uint8)
    return rgbs



def train(model, train_data, valid_data, epoch, model_save_path = './save_model/', image_save_path = './save_images/', model_name = 'fcn8'):
     # Hyper parameter
    learning_rate = 1e-2
    weight_decay = 3e-4
    momentum = 0.9

    print(f'Start training {model_name}......')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device used:', device)

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
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

        for batch_idx, (data, target, _) in enumerate(train_data):
            data, target = data.to(device), target.to(device,dtype=torch.long)
            optimizer.zero_grad()

            predict = model(data)
            predict = F.log_softmax(predict, dim=1)
        
            loss = criterion(predict, target)
            loss.backward()

            optimizer.step()

            label_pred = predict.max(dim=1)[1].data.cpu().numpy()
            target = target.cpu().numpy()

            # Record Loss & Acc
            train_loss += loss.item()

            if first:
                all_preds = label_pred
                all_labels = target
            else:
                all_preds = np.concatenate((all_preds,label_pred))
                all_labels = np.concatenate((all_labels,target))
            
            first = False
        
        train_loss = train_loss / len(train_data)
        train_miou = miou.mean_iou_score(all_preds,all_labels)

        print(f"[ Train | {ep:03d}/{epoch:03d} ] loss = {train_loss:.5f}, mean_iou = {train_miou:.5f}")

        # Validation
        model.eval()

        valid_loss = 0.0
        valid_miou = 0.0

        first = True

        for batch_idx, (data, target, img_indices) in enumerate(valid_data):
            data, target = data.to(device), target.to(device,dtype=torch.long)
            with torch.no_grad():
                predict = model(data)
                predict = F.log_softmax(predict, dim=1)

            loss = criterion(predict, target)
            
            label_pred = predict.max(dim=1)[1].data.cpu().numpy()
            target = target.cpu().numpy()

            if first:
                all_preds = label_pred
                all_labels = target
            else:
                all_preds = np.concatenate((all_preds,label_pred))
                all_labels = np.concatenate((all_labels,target))
            
            first = False
            valid_loss += loss

            if model_name == 'fcn8':
                if ep in [1,10,epoch]:
                    if img_indices[0] in ["0010","0097","0107"]:
                        output_img = mask_lable_to_rgb(label_pred)
                        imgio.imsave(os.path.join(image_save_path, img_indices[0] + "_epoch_" + str(ep) + ".png"), output_img[0])
        
        valid_loss = valid_loss / len(valid_data)
        valid_miou = miou.mean_iou_score(all_preds,all_labels)

        print(f"[ Valid | {ep:03d}/{epoch:03d} ] loss = {valid_loss:.5f}, mean_iou = {valid_miou:.5f}")
        # save the best model
        if valid_miou > best_acc:
            best_acc = valid_miou
            print('best acc: ',best_acc)
            model_save = model.state_dict()
            save_file = model_name + '.pth'
            torch.save(model_save, os.path.join(model_save_path,save_file))
            print('save model with mIoU:',best_acc)

def test(model, test_data, pretrained_path, save_path):
    print('-'*20)
    print('| Test set predict |')
    print('-'*20)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device used:', device)
    model.load_state_dict(torch.load(pretrained_path))
    model.to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    # Validation
    model.eval()

    valid_loss = 0.0
    valid_miou = 0.0

    first = True

    for (data, target, img_indices) in test_data:
        data, target = data.to(device), target.to(device,dtype=torch.long)

        with torch.no_grad():
            predict = model(data)
            predict = F.log_softmax(predict, dim=1)

            loss = criterion(predict, target)
            
            label_pred = predict.max(dim=1)[1].data.cpu().numpy()
            target = target.cpu().numpy()

            if first:
                all_preds = label_pred
                all_labels = target
                all_indices = img_indices
            else:
                all_preds = np.concatenate((all_preds,label_pred))
                all_labels = np.concatenate((all_labels,target))
                all_indices = np.concatenate((all_indices,img_indices))
            
            first = False
            valid_loss += loss
        
    valid_loss = valid_loss / len(test_data)
    valid_miou = miou.mean_iou_score(all_preds,all_labels)
    print(f"[ Test set | loss = {valid_loss:.5f}, mean_iou = {valid_miou:.5f} ]")
    output_imgs = mask_lable_to_rgb(all_preds)
    for i in range(len(output_imgs)):
        imgio.imsave(os.path.join(save_path, all_indices[i] + ".png"), output_imgs[i])

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
    images, labels, indices = dataiter.next()

    print('(Trainset) Image tensor in each batch:', images.shape, images.dtype)
    print('(Trainset) Label tensor in each batch:', labels.shape, labels.dtype)

    # model = VGG16_FCN32s(7)

    # train(model, trainset_loader, validset_loader, 30, model_name='fcn32')

    model = VGG16_FCN8s(7)
    # print(model)
    train(model, trainset_loader, validset_loader, 30, model_name='fcn8')
    # summary(VGG16_FCN32s(7).cuda(),(3,512,512))
    # summary(model.cuda(),(3,512,512))



if __name__ == '__main__':
    training()