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
import random
from PIL import Image
import torchvision.utils as vutils

from torchvision.models.inception import inception_v3
from torch.autograd import Variable
from scipy.stats import entropy

## Hyper parameters ##
image_size = 64
channel_size = 3
lv_size = 100
G_fs = 64
D_fs = 64
lr_G = 0.0002
lr_D = 0.0004
beta1 = 0.5
epochs = 600
save_model_dir = './save_models/'
manualSeed = 6
#######################

class p1(Dataset):
    def __init__(self, root):

        self.images = None
        self.labels = None
        self.filenames = []
        self.fileindices = []
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        filenames = glob.glob(os.path.join(root, '*.png'))
        self.filenames = sorted(filenames)
        for fn in self.filenames:
            fn = fn.replace(root,"")
            self.fileindices.append(fn.replace("/",""))
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        image_fn = self.filenames[index]
        image = Image.open(image_fn)
            
        image = self.transform(image)

        return image

    def __len__(self):
        return self.len

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(lv_size, G_fs * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_fs * 8),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),

            nn.ConvTranspose2d(G_fs * 8, G_fs * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_fs * 4),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),

            nn.ConvTranspose2d(G_fs * 4, G_fs * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_fs * 2),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            
            nn.ConvTranspose2d(G_fs * 2, G_fs, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_fs),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            
            nn.ConvTranspose2d(G_fs, channel_size, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channel_size, D_fs, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(D_fs, D_fs * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_fs * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(D_fs * 2, D_fs * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_fs * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(D_fs * 4, D_fs * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_fs * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(D_fs * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

## Ref from https://github.com/sbarratt/inception-score-pytorch ##
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

       
def training(data_root):

    # use fixed random seed
    print("Random Seed: ", manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    trainset = p1(root=data_root)

    print('# images in trainset:', len(trainset))

    trainset_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    dataiter = iter(trainset_loader)
    images = dataiter.next()
    print('(Trainset) Image tensor in each batch:', images.shape, images.dtype)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    G_model = Generator().to(device)
    G_model.apply(weights_init)

    D_model = Discriminator().to(device)
    D_model.apply(weights_init)

    criterion = nn.BCELoss()

    real_label = 1
    fake_label = 0

    G_optimizer = optim.Adam(G_model.parameters(), lr=lr_G, betas=(beta1, 0.999))
    D_optimizer = optim.Adam(D_model.parameters(), lr=lr_D, betas=(beta1, 0.999))

    G_losses = []
    D_losses = []

    data_size = len(trainset_loader)

    D_model.train()
    G_model.train()

    print("Start Training ...")
    for epoch in range(1, epochs+1):

        G_loss = 0.0
        D_loss = 0.0

        D_x_sum = 0.0
        D_G_z1_sum = 0.0
        D_G_z2_sum = 0.0

        for _, (data) in enumerate(trainset_loader):
            data = data.to(device)

            ##　Training Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            D_model.zero_grad()
            
            b_size = data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Train with real images
            output = D_model(data).view(-1)

            D_real_loss = criterion(output, label)
            D_real_loss.backward()
            
            # D(x)
            D_x = output.mean().item()
            D_x_sum += D_x

            ## Train with fake images
            noise = torch.randn(b_size, lv_size, 1, 1, device=device)

            fake = G_model(noise)
            label.fill_(fake_label)

            output = D_model(fake.detach()).view(-1)

            D_fake_loss = criterion(output, label)
            D_fake_loss.backward()

            # D(G(z)) before update D
            D_G_z1 = output.mean().item()
            D_G_z1_sum += D_G_z1
            # Discriminator loss = loss from training with real images + loss from training with fake images
            loss_D = D_real_loss + D_fake_loss

            D_loss += loss_D.item()
            # Update D optimizer
            D_optimizer.step()

            
            # Training Generator: maximize log(D(G(z)))
            G_model.zero_grad()
            label.fill_(real_label)

            output = D_model(fake).view(-1)

            loss_G = criterion(output, label)
            G_loss += loss_G.item()
            loss_G.backward()

            # D(G(z)) after update D
            D_G_z2 = output.mean().item()
            D_G_z2_sum += D_G_z2
            # Update G optimizer
            G_optimizer.step()

        G_loss = G_loss / data_size
        D_loss = D_loss / data_size

        D_x = D_x_sum / data_size
        D_G_z1 = D_G_z1_sum / data_size
        D_G_z2 = D_G_z2_sum / data_size

        print(f"[ Train | {epoch:03d}/{epochs:03d} ] G_loss = {G_loss:.4f}, D_loss = {D_loss:.4f}, D(x) = {D_x:.4f}, D(G(z)) before update D = {D_G_z1:.4f}, D(G(z)) after update D = {D_G_z2:.4f}")
        # Save Losses for plotting later
        G_losses.append(G_loss)
        D_losses.append(D_loss)

        G_model.eval()
        noise = torch.randn(1000, lv_size, 1, 1, device=device)

        i_s = inception_score(G_model(noise).detach().cpu(), cuda=True, resize=True, batch_size=50)[0]
        print(f'IS: {i_s:.4f}')

        if (D_loss <= 0.0001) or (D_loss >= 50.0):
            break
        
        # if inception score larger than strong baseline, then save model.
        if (epoch >= 100) and (i_s >= 2.15):
            torch.save(G_model.state_dict(), os.path.join(save_model_dir,f'g_model_bs128_flip_{epoch}_{i_s}.pth'))


def inference(model_path,save_img_path):

    print('-'*24)
    print('| Generating images... |')
    print('-'*24)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    G1 = Generator().to(device)
    G1.load_state_dict(torch.load(model_path))

    manualSeed = 6
    print("Random Seed: ", manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    G1.eval()
    noise = torch.randn(1000, lv_size, 1, 1, device=device)

    fake = G1(noise)

    for i, img in enumerate(fake):
        # imgio.imsave(os.path.join('./save_images', f'{i:04d}.jpg'), img)
        vutils.save_image(img, os.path.join(save_img_path, f'{i:04d}.png'), normalize=True)

    # i_s = inception_score(fake.detach().cpu(), cuda=True, resize=True, batch_size=50)[0]
    # print(f'IS: {i_s:.4f}')
    
    print(f'All images save in {save_img_path}')

if __name__ == '__main__':
    training('hw2_data/face/train')
