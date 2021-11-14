import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob


class p2(Dataset):
    def __init__(self, root):

        self.images = None
        self.labels = None
        self.filenames = []
        self.fileindices = []
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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

        label = int(self.fileindices[index][0])

        return image, label

    def __len__(self):
        return self.len

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    
    # load digit classifier
    net = Classifier()
    path = "Classifier.pth"
    load_checkpoint(path, net)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        net = net.to(device)

    print(net)
    dataset = p2('./save_digits')

    print('# images in dataset:', len(dataset))

    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    dataiter = iter(dataset_loader)
    images, labels = dataiter.next()
    print('(Trainset) Image tensor in each batch:', images.shape, images.dtype)
    print('(Trainset) Label tensor in each batch:', labels.shape, labels.dtype)

    cnt = 0
    size = len(dataset_loader)
    for (data,label) in dataset_loader:
        data = data.to(device)
        classes = net(data)
        if (int(classes[0].argmax(dim=-1).cpu().numpy()) == label):
            cnt += 1
    print(f'Accuracy: {cnt/size*100:.2f}%')

