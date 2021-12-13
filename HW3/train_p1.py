import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_pretrained_vit import ViT
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import glob
import os
import numpy as np
from PIL import Image
import pandas as pd
## Hyper parameters ##
image_size = 384
channel_size = 3
num_classes = 37
epochs = 30
save_model_dir = './models/'
#######################

class p1(Dataset):
    def __init__(self, root, target='train'):

        self.images = None
        self.labels = []
        self.filenames = []
        self.image_names = []
        self.root = root
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])

        if target == 'test':
            self.transform = transforms.Compose([
                transforms.Resize((image_size,image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5],[0.5])
            ])

        self.filenames = glob.glob(os.path.join(root, '*.jpg'))
        for fn in self.filenames:
            fn = fn.replace(root,"")
            fn = fn.replace("/","")
            self.image_names.append(fn)
            self.labels.append(fn.split('_')[0])
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        image_fn = self.filenames[index]
        image = Image.open(image_fn).convert('RGB')
            
        image = self.transform(image)
        label = self.labels[index]

        return image, int(label)

    def __len__(self):
        return self.len

class p1_inference(Dataset):
    def __init__(self, root):

        self.images = None
        self.filenames = []
        self.image_names = []
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])

        self.filenames = glob.glob(os.path.join(root, '*.jpg'))
        for fn in self.filenames:
            fn = fn.replace(root,"")
            fn = fn.replace("/","")
            self.image_names.append(fn)
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        image_fn = self.filenames[index]
        image = Image.open(image_fn).convert('RGB')
            
        image = self.transform(image)

        return image, self.image_names[index]

    def __len__(self):
        return self.len

def inference(model_path,images_path,output_path):

    testset = p1_inference(root=images_path)
    print('# images in testset:', len(testset))

    testset_loader = DataLoader(testset, batch_size=1, shuffle=False)

    model = ViT('L_16_imagenet1k', pretrained=True)
    model.fc = nn.Linear(1024,num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device used:', device)

    model.to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    preds = []
    img_names = []

    for (data, image_name) in tqdm(testset_loader):
        data = data.to(device)

        with torch.no_grad():
            predict = model(data)[0]
            preds.append(predict.argmax(dim=-1).cpu().numpy())
            img_name = image_name[0]
            img_names.append(img_name)
    
    output = pd.DataFrame({'filename':img_names,'label':preds})
    output.to_csv(output_path,index=False)
    print(f'Result save as "{output_path}"')

if __name__ == '__main__':
    trainset = p1(root='hw3_data/p1_data/train')
    validset = p1(root='hw3_data/p1_data/val',target='test')
    print('# images in trainset:', len(trainset))
    print('# images in validset:', len(validset))

    trainset_loader = DataLoader(trainset, batch_size=2, shuffle=True)
    validset_loader = DataLoader(validset, batch_size=2, shuffle=False)

    model = ViT('L_16_imagenet1k', pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(1024,512),
        nn.Dropout(),
        nn.LeakyReLU(0.2),
        nn.Linear(512,num_classes)
    )
    for param in model.fc.parameters():
        param.requires_grad = True

    # Hyper parameter
    learning_rate = 1e-4
    weight_decay = 3e-4
    momentum = 0.9

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device used:', device)

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay= weight_decay, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    
    model.train() 
    
    best_acc = 0.0
    model_save = model.state_dict()

    # Early stop
    patient = 3
    trigger = 0

    min_loss = np.inf
    # print(model)

    for epoch in range(1, epochs+1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)

        # train
        train_loss = 0.0
        train_acc = 0.0

        for (data, target) in tqdm(trainset_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            # Acc
            acc = (output.argmax(dim=-1) == target).float().mean()

            # Record Loss & Acc
            train_loss += loss
            train_acc += acc
        
        train_loss = train_loss / len(trainset_loader)
        train_acc = train_acc / len(trainset_loader)

        print(f"[ Train | {epoch:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # Validation
        model.eval()

        valid_loss = 0.0
        valid_acc = 0.0

        for (data, target) in tqdm(validset_loader):
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                predict = model(data)

            loss = criterion(predict, target)

            acc = (predict.argmax(dim=-1) == target).float().mean()
            
            valid_loss += loss
            valid_acc += acc
        
        valid_loss = valid_loss / len(validset_loader)
        valid_acc = valid_acc / len(validset_loader)

        print(f"[ Valid | {epoch:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        # save the best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('best acc: ',best_acc)
            if best_acc > 0.95:
                model_save = model.state_dict()
                save_file = f"vit_L16_{best_acc.item():.4f}.pth"
                torch.save(model_save, os.path.join(save_model_dir,save_file))
                print('save model with acc:',best_acc)

        if valid_loss < min_loss:
            min_loss = valid_loss
            trigger = 0
        else:
            trigger += 1
            if trigger >= patient:
                break
    # for _ in range(1):
    #     # Validation
    #     model.load_state_dict(torch.load('save_models/vit_0.9467.pth'))
    #     model.eval()

    #     valid_loss = 0.0
    #     valid_acc = 0.0

    #     for (data, target) in tqdm(validset_loader):
    #         data, target = data.to(device), target.to(device)

    #         with torch.no_grad():
    #             predict = model(data)

    #         loss = criterion(predict, target)

    #         acc = (predict.argmax(dim=-1) == target).float().mean()
            
    #         valid_loss += loss
    #         valid_acc += acc
        
    #     valid_loss = valid_loss / len(validset_loader)
    #     valid_acc = valid_acc / len(validset_loader)

    #     print(f"[ Valid | loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")