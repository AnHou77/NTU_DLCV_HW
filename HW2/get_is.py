import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import train_p1 as p1
import os

class dataset(Dataset):
    def __init__(self, root):

        self.images = None
        self.labels = None
        self.filenames = []
        self.fileindices = []
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((64,64)),
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

        return image

    def __len__(self):
        return self.len

if __name__ == '__main__':

    images = dataset('./save_images')
    is_score = p1.inception_score(images, cuda=True, resize=True, batch_size=50)[0]
    print(f'IS: {is_score:.4f}')