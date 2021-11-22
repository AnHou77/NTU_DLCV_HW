import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.autograd import Function

import glob
import os

from torchvision.transforms.autoaugment import AutoAugmentPolicy
import numpy as np
import pandas as pd
from PIL import Image

## Hyper parameters ##
image_size = 28
image_size_64x = 64
channel_size = 3
num_classes = 10
lr = 1e-3
weight_decay= 3e-4
epochs = 100
batch_size=128
save_model_dir = './save_models/'
#######################

class p3(Dataset):
    def __init__(self, root, img_size, label_path=None, target=None):

        self.images = None
        self.labels = []
        self.filenames = []
        self.fileindices = []
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        if target=='svhn':
            self.transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.AutoAugment(AutoAugmentPolicy.SVHN),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        filenames = glob.glob(os.path.join(root, '*.png'))
        self.filenames = sorted(filenames)
        for fn in self.filenames:
            fn = fn.replace(root,"")
            self.fileindices.append(fn.replace("/",""))
        
        if label_path is not None:
            labeltable = pd.read_csv(label_path)
            labels = labeltable['label'].to_numpy()
            for i in range(len(self.fileindices)):
                self.labels.append(labels[i])
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        image_fn = self.filenames[index]
        image = Image.open(image_fn).convert('RGB')
            
        image = self.transform(image)

        label = 0
        if len(self.labels) > 0:
            label = self.labels[index]
        return image, label

    def __len__(self):
        return self.len

class ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, lambdaa):
        ctx.lambdaa = lambdaa
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.neg() * ctx.lambdaa, None

class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(channel_size, image_size, 4, 1, 0),
            nn.BatchNorm2d(image_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(image_size, image_size * 2, 3, 2, 1),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(image_size * 2, image_size * 4, 3, 2, 1),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(image_size * 4, image_size * 8, 3, 2, 1),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(image_size * 8, image_size * 16, 4, 1, 0),
            nn.BatchNorm2d(image_size * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )
        self.classes_classifier = nn.Sequential(
            nn.Linear(image_size * 16, image_size * 8),
            nn.ReLU(inplace=True),
            nn.Linear(image_size * 8, num_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(image_size * 16, image_size * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(image_size * 8, image_size * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(image_size * 4, 1),
            nn.Sigmoid()
        )
    def forward(self, x, lambdaa=0):
        x = self.feature_extractor(x)
        x = x.view(-1, image_size * 16)
        reverse_feature = ReverseLayer.apply(x, lambdaa)
        return self.classes_classifier(x), self.domain_classifier(reverse_feature)

class DANN_64x(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(channel_size, image_size_64x, 4, 1, 0),
            nn.BatchNorm2d(image_size_64x),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),

            nn.Conv2d(image_size_64x, image_size_64x * 2, 3, 2, 1),
            nn.BatchNorm2d(image_size_64x * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),

            nn.Conv2d(image_size_64x * 2, image_size_64x * 4, 3, 2, 1),
            nn.BatchNorm2d(image_size_64x * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),

            nn.Conv2d(image_size_64x * 4, image_size_64x * 8, 3, 2, 1),
            nn.BatchNorm2d(image_size_64x * 8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),

            nn.Conv2d(image_size_64x * 8, image_size_64x * 16, 3, 2, 1),
            nn.BatchNorm2d(image_size_64x * 16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),

            nn.Conv2d(image_size_64x * 16, image_size_64x * 32, 4, 1, 0),
            nn.BatchNorm2d(image_size_64x * 32),
            nn.ReLU(inplace=True),
        )
        self.classes_classifier = nn.Sequential(
            nn.Linear(image_size_64x * 32, image_size_64x * 16),
            nn.ReLU(inplace=True),
            nn.Linear(image_size_64x * 16, image_size_64x * 8),
            nn.ReLU(inplace=True),
            nn.Linear(image_size_64x * 8, image_size_64x * 4),
            nn.ReLU(inplace=True),
            nn.Linear(image_size_64x * 4, image_size_64x * 2),
            nn.ReLU(inplace=True),
            nn.Linear(image_size_64x * 2, num_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(image_size_64x * 32, image_size_64x * 16),
            nn.LeakyReLU(0.2,True),
            nn.Linear(image_size_64x * 16, image_size_64x * 8),
            nn.LeakyReLU(0.2,True),
            nn.Linear(image_size_64x * 8, image_size_64x * 4),
            nn.LeakyReLU(0.2,True),
            nn.Linear(image_size_64x * 4, 1),
            nn.Sigmoid()
        )
    def forward(self, x, lambdaa=0):
        x = self.feature_extractor(x)
        x = x.view(-1, image_size_64x * 32)
        reverse_feature = ReverseLayer.apply(x, lambdaa)
        return self.classes_classifier(x), self.domain_classifier(reverse_feature)

def train(model, src_trainset_loader, target_trainset_loader, save_model_name):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay= weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay= 3e-4, momentum=0.9)

    criterion_domain = nn.BCELoss()
    criterion_classes = nn.CrossEntropyLoss()

    real_label = 1
    fake_label = 0

    dataloader_size = len(src_trainset_loader)

    src_classes_losses = []
    src_domain_losses = []
    target_domain_losses = []
    model_losses = []
    acc_arr = []
    best_acc = 0.0

    print("Start Training ...")
    for epoch in range(1, epochs+1):
        model.train()
        src_loader = iter(src_trainset_loader)
        target_loader = iter(target_trainset_loader)
        src_classes_loss_total = 0.0
        src_domain_loss_total = 0.0
        target_domain_loss_total = 0.0
        model_loss = 0.0
        accs = 0.0
        size = 0
        for i in range(dataloader_size):
            lambdaa = 2. / (1. + np.exp(-10 * float(i + (epoch-1) * dataloader_size / (epochs * dataloader_size)))) - 1.

            model.zero_grad()
            # training source data
            src_imgs, src_labels = src_loader.next()
            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)

            b_size = src_imgs.size(0)

            classes_cf, domain_cf = model(src_imgs, lambdaa)
            src_classes_loss = criterion_classes(classes_cf, src_labels)
            src_classes_loss_total += src_classes_loss.item()
            domain_label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            src_domain_loss = criterion_domain(domain_cf.view(-1), domain_label)
            src_domain_loss_total += src_domain_loss.item()

            acc = 0
            for x, cs in enumerate(classes_cf):
                class_label = int(classes_cf[x].argmax(dim=-1).cpu().numpy())
                if class_label == src_labels[x]:
                    acc += 1
            
            accs += acc
            size += b_size

            # training target data
            target_imgs, _ = target_loader.next()
            target_imgs = target_imgs.to(device)

            b_size = target_imgs.size(0)

            _, domain_cf = model(target_imgs, lambdaa)
            domain_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            target_domain_loss = criterion_domain(domain_cf.view(-1), domain_label)
            target_domain_loss_total += target_domain_loss.item()

            loss = src_classes_loss + 0.5  * src_domain_loss + 0.5 * target_domain_loss
            model_loss += loss.item()
            loss.backward()
            optimizer.step()

        src_classes_loss_total /= dataloader_size
        src_domain_loss_total /= dataloader_size
        target_domain_loss_total /= dataloader_size
        model_loss /= dataloader_size
        # accs /= dataloader_size
        accs /= size

        print(f'[ Train | {epoch:03d}/{epochs:03d} ] Accuracy = {accs:.4f}, src_classes_loss = {src_classes_loss_total:.4f}, src_domain_loss = {src_domain_loss_total:.4f}, target_domain_loss = {target_domain_loss_total:.4f}, all_loss = {model_loss:.4f}')

        src_classes_losses.append(src_classes_loss_total)
        src_domain_losses.append(src_domain_loss_total)
        target_domain_losses.append(target_domain_loss_total)
        model_losses.append(model_loss)
        acc_arr.append(accs)

        accs = 0.0
        size = 0
        model.eval()
        for _, (target_imgs, target_labels) in enumerate(target_trainset_loader):
            target_imgs, target_labels = target_imgs.to(device), target_labels.to(device)
            lambdaa = 2. / (1. + np.exp(-10 * float(i + (epoch-1) * dataloader_size / (epochs * dataloader_size)))) - 1.

            b_size = target_labels.size(0)
            classes_cf, _ = model(target_imgs, lambdaa)
            
            acc = 0
            for i, cs in enumerate(classes_cf):
                class_label = int(classes_cf[i].argmax(dim=-1).cpu().numpy())
                if class_label == target_labels[i]:
                    acc += 1
            size += b_size
            accs += acc
        accs /= size
        if accs > best_acc:
            best_acc = accs
            torch.save(model.state_dict(), os.path.join(save_model_dir, save_model_name))
        print(f'[ Valid | {epoch:03d}/{epochs:03d} ] Accuracy = {accs:.4f}')
    print(f'Best accuracy: {best_acc:.2f}')

def test(model_val, target_testset_loader, load_model_name):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model_val = model_val.to(device)
    model_val.load_state_dict(torch.load(os.path.join(save_model_dir,load_model_name)))
    accs = 0
    size = 0
    model_val.eval()
    for _, (target_imgs, target_labels) in enumerate(target_testset_loader):
        target_imgs, target_labels = target_imgs.to(device), target_labels.to(device)
        lambdaa = 0

        b_size = target_labels.size(0)
        classes_cf, _ = model_val(target_imgs, lambdaa)
        acc = 0
        for i, cs in enumerate(classes_cf):
            class_label = int(classes_cf[i].argmax(dim=-1).cpu().numpy())
            if class_label == target_labels[i]:
                acc += 1
        size += b_size
        accs += acc
    # accs /= size
    print(f'[ Test | Accuracy = {accs / size:.4f} ({accs} / {size})]')

def train_on_one_domain(model, src_trainset_loader, validset_loader, save_model_name):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion_classes = nn.CrossEntropyLoss()

    dataloader_size = len(src_trainset_loader)

    best_acc = 0.0

    print("Start Training ...")
    for epoch in range(1, epochs+1):
        model.train()
        src_loader = iter(src_trainset_loader)
        src_classes_loss_total = 0.0
        accs = 0.0
        size = 0
        for i in range(dataloader_size):
            lambdaa = 2. / (1. + np.exp(-10 * float(i + (epoch-1) * dataloader_size / (epochs * dataloader_size)))) - 1.

            model.zero_grad()
            # training source data
            src_imgs, src_labels = src_loader.next()
            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)

            b_size = src_imgs.size(0)

            classes_cf, _ = model(src_imgs, lambdaa)
            src_classes_loss = criterion_classes(classes_cf, src_labels)
            src_classes_loss_total += src_classes_loss.item()

            src_classes_loss.backward()
            optimizer.step()

            acc = 0
            for x, _ in enumerate(classes_cf):
                class_label = int(classes_cf[x].argmax(dim=-1).cpu().numpy())
                if class_label == src_labels[x]:
                    acc += 1
            
            accs += acc
            size += b_size

        src_classes_loss_total /= dataloader_size
        accs /= size

        print(f'[ Train | {epoch:03d}/{epochs:03d} ] Accuracy = {accs:.4f}, src_classes_loss = {src_classes_loss_total:.4f}')

        accs = 0.0
        size = 0
        model.eval()
        for _, (target_imgs, target_labels) in enumerate(validset_loader):
            target_imgs, target_labels = target_imgs.to(device), target_labels.to(device)
            lambdaa = 2. / (1. + np.exp(-10 * float(i + (epoch-1) * dataloader_size / (epochs * dataloader_size)))) - 1.

            b_size = target_labels.size(0)
            classes_cf, _ = model(target_imgs, lambdaa)
            
            acc = 0
            for i, cs in enumerate(classes_cf):
                class_label = int(classes_cf[i].argmax(dim=-1).cpu().numpy())
                if class_label == target_labels[i]:
                    acc += 1
            size += b_size
            accs += acc
        accs /= size

        if accs > best_acc:
            best_acc = accs
            torch.save(model.state_dict(), os.path.join(save_model_dir, save_model_name))
        print(f'[ Valid | {epoch:03d}/{epochs:03d} ] Accuracy = {accs:.4f}')

def train_source_and_target(model, src_trainset, src_testset, target_trainset, target_testset, save_model_name):
    train_data_size = range(min(len(src_trainset), len(target_trainset)))

    src_trainset = Subset(src_trainset, train_data_size)
    target_trainset = Subset(target_trainset,train_data_size)

    print('# images(filtered) in source trainset:', len(src_trainset))
    print('# images(filtered) in target trainset:', len(target_trainset))

    src_trainset_loader = DataLoader(src_trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    src_testset_loader = DataLoader(src_testset, batch_size=batch_size, shuffle=False, num_workers=4)
    target_trainset_loader = DataLoader(target_trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    target_testset_loader = DataLoader(target_testset, batch_size=batch_size, shuffle=False, num_workers=4)

    dataiter = iter(src_trainset_loader)
    images, labels = dataiter.next()
    print('(Source Trainset) Image tensor in each batch:', images.shape, images.dtype)
    print('(Source Trainset) Label tensor in each batch:', labels.shape, labels.dtype)
    dataiter = iter(target_trainset_loader)
    images, labels = dataiter.next()
    print('(Target Trainset) Image tensor in each batch:', images.shape, images.dtype)
    print('(Target Trainset) Label tensor in each batch:', labels.shape, labels.dtype)

    train(model=model, src_trainset_loader=src_trainset_loader, target_trainset_loader=target_trainset_loader, save_model_name=save_model_name)
    test(model,target_testset_loader=target_testset_loader,load_model_name=save_model_name)
    
def train_only_source(model, src_trainset, src_testset, target_trainset, target_testset, save_model_name):
    print('# images(filtered) in source trainset:', len(src_trainset))
    print('# images(filtered) in target trainset:', len(target_trainset))

    src_trainset_loader = DataLoader(src_trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    src_testset_loader = DataLoader(src_testset, batch_size=batch_size, shuffle=False, num_workers=4)
    target_trainset_loader = DataLoader(target_trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    target_testset_loader = DataLoader(target_testset, batch_size=batch_size, shuffle=False, num_workers=4)

    dataiter = iter(src_trainset_loader)
    images, labels = dataiter.next()
    print('(Source Trainset) Image tensor in each batch:', images.shape, images.dtype)
    print('(Source Trainset) Label tensor in each batch:', labels.shape, labels.dtype)

    train_only_source(model=model, src_trainset_loader=src_trainset_loader, validset_loader=target_trainset_loader, save_model_name=save_model_name)
    test(model,target_testset_loader=src_testset_loader,load_model_name=save_model_name)
    test(model,target_testset_loader=target_testset_loader,load_model_name=save_model_name)

def inference(model_path,target_data_path,target_domain_name,output_path):
    if target_domain_name == 'svhn':
        img_size = 64
        model = DANN_64x()
    else:
        img_size = 28
        model = DANN()

    target_testset = p3(root=target_data_path,img_size=img_size)
    print('# images in target testset:', len(target_testset))
    target_testset_loader = DataLoader(target_testset, batch_size=1, shuffle=False, num_workers=1)
    dataiter = iter(target_testset_loader)
    images, _ = dataiter.next()
    print('(Target Testset) Image tensor in each batch:', images.shape, images.dtype)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model_val = model.to(device)
    model_val.load_state_dict(torch.load(model_path))
    model_val.eval()
    image_names = []
    labels = []
    for i, (target_imgs, _) in enumerate(target_testset_loader):
        target_imgs = target_imgs.to(device)

        classes_cf, _ = model_val(target_imgs)
        for cf in classes_cf:
            class_label = int(cf.argmax(dim=-1).cpu().numpy())
            labels.append(class_label)
            image_names.append(f'{i:05d}.png')
            
    result = pd.DataFrame({'image_name':image_names, 'label':labels})
    result.to_csv(output_path,index=False)
    print(f'Result(csv) save in {output_path}')
if __name__ == '__main__':
    ### Example ###

    ## train usps
    # src_trainset = p3(root='hw2_data/digits/usps/train',label_path='hw2_data/digits/usps/train.csv',img_size=64, target='svhn')
    # src_testset = p3(root='hw2_data/digits/usps/test',label_path='hw2_data/digits/usps/test.csv',img_size=64)
    # target_trainset = p3(root='hw2_data/digits/svhn/train',label_path='hw2_data/digits/svhn/train.csv',img_size=64)
    # target_testset = p3(root='hw2_data/digits/svhn/test',label_path='hw2_data/digits/svhn/test.csv',img_size=64)

    # print('# images in source trainset:', len(src_trainset))
    # print('# images in source testset:', len(src_testset))
    # print('# images in target trainset:', len(target_trainset))
    # print('# images in target testset:', len(target_testset))
    # model = DANN_64x()
    # train_source_and_target(model,src_trainset,src_testset,target_trainset,target_testset,'dann_usps_svhn_64.pth')

    # train mnistm or svhn
    # src_trainset = p3(root='hw2_data/digits/mnistm/train',img_size=28,label_path='hw2_data/digits/mnistm/train.csv', target='svhn')
    # src_testset = p3(root='hw2_data/digits/mnistm/test',img_size=28,label_path='hw2_data/digits/mnistm/test.csv')
    # target_trainset = p3(root='hw2_data/digits/usps/train',img_size=28,label_path='hw2_data/digits/usps/train.csv', target='svhn')
    # target_testset = p3(root='hw2_data/digits/usps/test',img_size=28,label_path='hw2_data/digits/usps/test.csv')

    # print('# images in source trainset:', len(src_trainset))
    # print('# images in source testset:', len(src_testset))
    # print('# images in target trainset:', len(target_trainset))
    # print('# images in target testset:', len(target_testset))
    # model = DANN()
    # train_source_and_target(model,src_trainset,src_testset,target_trainset,target_testset,'dann_mnistm_usps_28.pth')
    print(DANN(),DANN_64x())