import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary.torchsummary import summary
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import glob
import os
import numpy as np
import pandas as pd
import random
from PIL import Image
import torchvision.utils as vutils
from tqdm import tqdm

from torchsummary import summary
import digit_classifier as dc

## Hyper parameters ##
image_size = 28
lv_size = 100
channel_size = 3
G_fs = 28
D_fs = 28
num_classes = 10
lr_G = 0.0002
lr_D = 0.00025
beta1 = 0.5
epochs = 500
batch_size=256
save_model_dir = './save_models/'
manualSeed = 86
#######################

class p2(Dataset):
    def __init__(self, root, label_path):

        self.images = None
        self.labels = []
        self.filenames = []
        self.fileindices = []
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        filenames = glob.glob(os.path.join(root, '*.png'))
        self.filenames = sorted(filenames)
        for fn in self.filenames:
            fn = fn.replace(root,"")
            self.fileindices.append(fn.replace("/",""))
        
        labeltable = pd.read_csv(label_path)
        labels = labeltable['label'].to_numpy()
        for i in range(len(self.fileindices)):
            self.labels.append(labels[i])
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        image_fn = self.filenames[index]
        image = Image.open(image_fn)
            
        image = self.transform(image)

        label = self.labels[index]

        return image, label

    def __len__(self):
        return self.len

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(lv_size, G_fs * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_fs * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(G_fs * 8, G_fs * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(G_fs * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(G_fs * 4, G_fs * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(G_fs * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(G_fs * 2, G_fs, 3, 2, 1, bias=False),
            nn.BatchNorm2d(G_fs),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(G_fs, channel_size, 4, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channel_size, D_fs, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(D_fs, D_fs * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(D_fs * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(D_fs * 2, D_fs * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(D_fs * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(D_fs * 4, D_fs * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(D_fs * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(D_fs * 8, D_fs, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )

        self.fc_real_fake = nn.Linear(D_fs, 1)
        self.fc_classes = nn.Linear(D_fs, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, D_fs)
        return self.sigmoid(self.fc_real_fake(x)).view(-1), self.softmax(self.fc_classes(x))

    

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def train(train_data_path,label_path):
    trainset = p2(root=train_data_path,label_path=label_path)

    print('# images in trainset:', len(trainset))

    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    dataiter = iter(trainset_loader)
    images, labels = dataiter.next()
    print('(Trainset) Image tensor in each batch:', images.shape, images.dtype)
    print('(Trainset) Label tensor in each batch:', labels.shape, labels.dtype)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    G_model = Generator().to(device)
    G_model.apply(weights_init)

    D_model = Discriminator().to(device)
    D_model.apply(weights_init)

    net = dc.Classifier()
    path = "Classifier.pth"
    dc.load_checkpoint(path, net)
    net = net.to(device)

    criterion_real_fake = nn.BCELoss()
    criterion_classes = nn.CrossEntropyLoss()

    real_label = 1
    fake_label = 0

    G_optimizer = optim.Adam(G_model.parameters(), lr=lr_G, betas=(beta1, 0.999))
    D_optimizer = optim.Adam(D_model.parameters(), lr=lr_D, betas=(beta1, 0.999))

    G_losses_real_fake = []
    D_losses_real_fake = []
    G_losses_classes = []
    D_losses_classes = []

    data_size = len(trainset_loader)

    best_acc = 0.0
    best_epoch = 0

    print("Start Training ...")
    for epoch in range(1, epochs+1):

        G_loss_real_fake_total = 0.0
        D_loss_real_fake_total = 0.0
        G_loss_classes_total = 0.0
        D_loss_classes_total = 0.0

        for _, (data, label_classes) in enumerate(tqdm(trainset_loader)):
            data, label_classes = data.to(device), label_classes.to(device)

            # Discriminator training with real images
            D_optimizer.zero_grad()

            b_size = data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            real_fake, classes = D_model(data)
            D_loss_real_fake = criterion_real_fake(real_fake,label)
            D_loss_classes = criterion_classes(classes, label_classes)

            D_loss_real_fake_total += D_loss_real_fake.item()
            D_loss_classes_total += D_loss_classes.item()

            D_real_loss = D_loss_real_fake + D_loss_classes
            D_real_loss.backward()

            # Discriminator training with fake images
            label_ = np.random.randint(0, num_classes, b_size)
            noise_ = np.random.normal(0, 1, (b_size, lv_size))
            label_onehot = np.zeros((b_size, num_classes))
            label_onehot[np.arange(b_size), label_] = 1
            noise_[np.arange(b_size), :num_classes] = label_onehot[np.arange(b_size)]
            
            noise_ = (torch.from_numpy(noise_))
            noise_ = noise_.resize_(b_size, lv_size, 1, 1)
            noise = torch.randn(b_size, lv_size, 1, 1, device=device)
            noise.data.copy_(noise_)
            fake_classes_label = torch.randint(0,10,(b_size,), device=device)
            fake_classes_label.data.resize_(b_size).copy_(torch.from_numpy(label_))
            fake_imgs = G_model(noise)
            label.fill_(fake_label)
            real_fake, classes = D_model(fake_imgs.detach())
            D_loss_real_fake = criterion_real_fake(real_fake,label)

            D_loss_classes = criterion_classes(classes, fake_classes_label)

            D_loss_real_fake_total += D_loss_real_fake.item()
            D_loss_classes_total += D_loss_classes.item()

            D_fake_loss = D_loss_real_fake + D_loss_classes
            D_fake_loss.backward()
                
            D_optimizer.step()

            ## Generator training
            G_model.zero_grad()
            label.fill_(real_label)

            real_fake, classes = D_model(fake_imgs)
            G_loss_real_fake = criterion_real_fake(real_fake,label)
            G_loss_classes = criterion_classes(classes, fake_classes_label)

            G_loss_real_fake_total += G_loss_real_fake.item()
            G_loss_classes_total += G_loss_classes.item()

            G_loss = G_loss_real_fake + G_loss_classes
            G_loss.backward()

            G_optimizer.step()
            
        D_loss_real_fake_total /= data_size
        D_loss_classes_total /= data_size
        G_loss_real_fake_total /= data_size
        G_loss_classes_total /= data_size
        
        # valid Generator performance\
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        size = 100
        cnt = 0
        G_model.eval()
        for i in range(10):
            labels = np.random.randint(i, i+1, size)
            noises = np.random.normal(0, 1, (size, lv_size))
            label_onehot = np.zeros((size, num_classes))
            label_onehot[np.arange(size), labels] = 1
            noises[np.arange(size), :num_classes] = label_onehot[np.arange(size)]
            noise_ = (torch.from_numpy(noises))
            noise_ = noise_.resize_(size, lv_size, 1, 1)
            noise = torch.randn(size, lv_size, 1, 1, device=device)
            noise.data.copy_(noise_)
            
            output = G_model(noise)

            classes = net(output)
            for x, img in enumerate(output):
                class_label = int(classes[x].argmax(dim=-1).cpu().numpy())
                if class_label == i:
                    cnt += 1
        
        acc = cnt / 1000 * 100
        if acc > best_acc:
            best_acc = acc
            torch.save(G_model.state_dict(), os.path.join(save_model_dir,'G_model.pth'))
            torch.save(D_model.state_dict(), os.path.join(save_model_dir,'D_model.pth'))
            best_epoch = epoch
            
            
        print(f"[ Train | {epoch:03d}/{epochs:03d} ] Accuracy = {acc:.2f}%, D_loss_real_fake = {D_loss_real_fake_total:.4f}, D_loss_classes = {D_loss_classes_total:.4f}, G_loss_real_fake = {G_loss_real_fake_total:.4f}, G_loss_classes = {G_loss_classes_total:.4f}")
            
        D_losses_real_fake.append(D_loss_real_fake_total)
        D_losses_classes.append(D_loss_classes_total)
        G_losses_real_fake.append(G_loss_real_fake_total)
        G_losses_classes.append(G_loss_classes_total)
    
    return best_acc, best_epoch, D_losses_real_fake, D_loss_classes, G_losses_real_fake, G_losses_classes


def inference(model_path,save_img_path):

    print('-'*24)
    print('| Generating images... |')
    print('-'*24)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    G_model = Generator().to(device)
    G_model.load_state_dict(torch.load(model_path))

    print("Random Seed: ", manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    size = 100
    for i in range(10):
        labels = np.random.randint(i, i+1, size)
        noises = np.random.normal(0, 1, (size, lv_size))
        label_onehot = np.zeros((size, num_classes))
        label_onehot[np.arange(size), labels] = 1
        noises[np.arange(size), :num_classes] = label_onehot[np.arange(size)]
        noise_ = (torch.from_numpy(noises))
        noise_ = noise_.resize_(size, lv_size, 1, 1)
        noise = torch.randn(size, lv_size, 1, 1, device=device)
        noise.data.copy_(noise_)

        G_model.eval()
        output = G_model(noise)

        for x, img in enumerate(output):
            img_name = f'{i}_{x+1:03d}.png'
            vutils.save_image(img, os.path.join(save_img_path, img_name), normalize=True)
    
    print(f'All images save in {save_img_path}')

if __name__ == '__main__':
    train()