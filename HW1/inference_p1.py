from train_p1 import resnet, test
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import sys
import glob
import os
from PIL import Image

class p1(Dataset):
    def __init__(self, root):

        self.images = None
        self.labels = None
        self.filenames = []
        self.fileindices = []
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # read filenames
        filenames = glob.glob(os.path.join(root, f'*.png'))
        self.filenames = sorted(filenames)
        self.img_indices = []

        for fn in self.filenames:
            fn = fn.replace(root,"")
            self.img_indices.append(fn.replace("/",""))
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        image_fn= self.filenames[index]
        image = Image.open(image_fn)
        image = self.transform(image)

        img_index = self.img_indices[index]

        return image, img_index

    def __len__(self):
        return self.len

if __name__ == '__main__':
    # load the testset

    data_path = sys.argv[1]
    save_path = sys.argv[2]

    test_set = p1(root=data_path)

    print('# images in testset:', len(test_set))

    # Use the torch dataloader to iterate through the dataset
    testset_loader = DataLoader(test_set, batch_size= 1, shuffle=False)

    # get some random training images
    dataiter = iter(testset_loader)
    images, indices = dataiter.next()

    print('Image tensor in each batch:', images.shape, images.dtype)

    model = resnet(50)
    # print(model)
    test(model, testset_loader, pretrained_path='./resnet152_0.8740.pth', save_path=save_path)