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
from PIL import Image
from tqdm import tqdm


class p1(Dataset):
    def __init__(self, root, transform=None):

        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        for i in range(50):
            filenames = glob.glob(os.path.join(root, f'{i}_*.png'))
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

class vgg16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = torchvision.models.vgg16(pretrained=True).to(device)

        self.model.classifier[6] = nn.Linear(4096,50)
        for param in self.model.parameters():
            param.requires_grad = True

        # # unfreeze the Linear layer, because it need to be trained.
        # for param in self.model._fc.parameters():
        #     param.requires_grad = True
        

    def forward(self, x):
        pred = self.model(x)
        return pred


def train(model, train_data, test_data, epoch, save_path = './save_model/'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device used:', device)

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()  # Important: set training mode
    
    best_acc = 0.0
    model_save = model.state_dict()

    for ep in range(1, epoch+1):
        print('Epoch {}/{}'.format(ep, epoch))
        print('-' * 10)

        # train
        train_loss = 0.0
        train_acc = 0.0
        for (data, target) in tqdm(train_data):
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

        for (data, target) in tqdm(test_data):
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                predict = model(data)

            loss = criterion(predict, target)

            acc = (predict.argmax(dim=-1) == target).float().mean()
            
            valid_loss += loss
            valid_acc += acc
        
        valid_loss = valid_loss / len(test_data)
        valid_acc = valid_acc / len(test_data)

        # Print the information.das
        print(f"[ Valid | {ep:03d}/{epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        # save the best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('best acc: ',best_acc)
            model_save = model.state_dict()

    save_file = f"vgg16_{best_acc.item():.4f}.pth"
    torch.save(model_save, os.path.join(save_path,save_file))
    print('save model with acc:',best_acc)
            
            

# load the testset
trainset = p1(root='data/p1_data/train_50', transform=transforms.ToTensor())
# load the testset
test_set = p1(root='data/p1_data/val_50', transform=transforms.ToTensor())

print('# images in trainset:', len(trainset))
print('# images in testset:', len(test_set))

# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
testset_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

# get some random training images
dataiter = iter(trainset_loader)
images, labels = dataiter.next()

print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)

model = vgg16()

train(model, trainset_loader, testset_loader, 100)