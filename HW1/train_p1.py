import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import glob
import os
import numpy as np
import pandas as pd
from PIL import Image

class p1(Dataset):
    def __init__(self, root, target):

        self.images = None
        self.labels = None
        self.filenames = []
        self.fileindices = []
        self.root = root
        if target == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.RandomChoice([
                        transforms.Pad(round(224 * 0.1)),
                        transforms.RandomCrop((round(224 * 0.9), round(224 * 0.9))),
                    ]),
                    transforms.Resize((224, 224))
                ], p=0.6),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # read filenames
        for i in range(50):
            filenames = glob.glob(os.path.join(root, f'{i}_*.png'))
            for fn in filenames:
                self.filenames.append((fn, i))
                self.fileindices.append(fn.replace(root,""))
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.len

class resnet(nn.Module):
    def __init__(self, num_class=50,pretrained_path=None):
        super().__init__()
        
        self.model = torchvision.models.resnet152(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048,1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(1024,num_class)
        )

        if pretrained_path is not None:
            self.model.state_dict(torch.load(pretrained_path))

        for param in self.model.parameters():
            param.requires_grad = True

        # for param in self.model.fc.parameters():
        #     param.requires_grad = True


    def forward(self, x):
        pred = self.model(x)
        return pred


def train(model, train_data, valid_data, epoch, save_path = './save_model/'):
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

    for ep in range(1, epoch+1):
        print('Epoch {}/{}'.format(ep, epoch))
        print('-' * 10)

        # train
        train_loss = 0.0
        train_acc = 0.0

        for batch_idx, (data, target) in enumerate(train_data):
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
        
        train_loss = train_loss / len(train_data)
        train_acc = train_acc / len(train_data)

        print(f"[ Train | {ep:03d}/{epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # Validation
        model.eval()

        valid_loss = 0.0
        valid_acc = 0.0

        for (data, target) in valid_data:
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                predict = model(data)

            loss = criterion(predict, target)

            acc = (predict.argmax(dim=-1) == target).float().mean()
            
            valid_loss += loss
            valid_acc += acc
        
        valid_loss = valid_loss / len(valid_data)
        valid_acc = valid_acc / len(valid_data)

        print(f"[ Valid | {ep:03d}/{epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        # save the best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('best acc: ',best_acc)
            model_save = model.state_dict()
            save_file = f"resnet152_{best_acc.item():.4f}.pth"
            torch.save(model_save, os.path.join(save_path,save_file))
            print('save model with acc:',best_acc)

        if valid_loss < min_loss:
            min_loss = valid_loss
            trigger = 0
        else:
            trigger += 1
            if trigger >= patient:
                break
            
def test(model, test_data, image_ids, pretrained_path, save_path):
    def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook
    print('-'*20)
    print('| Test set predict |')
    print('-'*20)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device used:', device)
    model.load_state_dict(torch.load(pretrained_path))
    model.to(device)

    model.model.fc[3].register_forward_hook(get_features('feats'))

    criterion = nn.CrossEntropyLoss()
    model.eval()

    test_loss = 0.0
    test_acc = 0.0

    preds = []
    FEATS = []
    first = True

    for (data, target) in test_data:
        data, target = data.to(device), target.to(device)

        features = {}

        with torch.no_grad():
            predict = model(data)

            loss = criterion(predict, target)

            # argmax
            pred = predict.argmax(dim=-1)

            preds += list(pred.cpu().numpy())
            acc = (pred == target).float().mean()
                
            test_loss += loss
            test_acc += acc             

        # FEATS.append()
        if first:
            FEATS = features['feats'].cpu().numpy()
        else:
            FEATS = np.concatenate((FEATS,features['feats'].cpu().numpy()))
        
        first = False
            
        
    test_loss = test_loss / len(test_data)
    test_acc = test_acc / len(test_data)

    print(f"Test set | loss = {test_loss:.5f}, acc = {test_acc:.5f}")

    output = pd.read_csv('data/p1_data/val_gt.csv')
    output['image_id'] = image_ids
    output['label'] = preds
    output.to_csv(save_path,index=False)
    print(f'Result save as "{save_path}"')
    return FEATS
def training():
    # load the trainset
    trainset = p1(root='data/p1_data/train_50', target= 'train')
    # load the validset
    validset = p1(root='data/p1_data/val_50', target= 'test')

    print('# images in trainset:', len(trainset))
    print('# images in validset:', len(validset))

    # Use the torch dataloader to iterate through the dataset
    trainset_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    validset_loader = DataLoader(validset, batch_size=64, shuffle=False, num_workers=4)

    # get some random training images
    dataiter = iter(trainset_loader)
    images, labels = dataiter.next()

    print('(Trainset) Image tensor in each batch:', images.shape, images.dtype)
    print('(Trainset) Label tensor in each batch:', labels.shape, labels.dtype)

    model = resnet(50)
    train(model, trainset_loader, validset_loader, 50)

if __name__ == '__main__':
    # training()
    pass