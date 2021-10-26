from train import p1, resnet, test
from torch.utils.data import DataLoader
import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

if __name__ == '__main__':
    # load the testset

    data_path = sys.argv[1]
    save_path = sys.argv[2]

    test_set = p1(root=data_path, target= 'test')

    print('# images in testset:', len(test_set))

    # Use the torch dataloader to iterate through the dataset
    testset_loader = DataLoader(test_set, batch_size= 25, shuffle=False, num_workers=4)

    # get some random training images
    dataiter = iter(testset_loader)
    images, labels = dataiter.next()

    print('Image tensor in each batch:', images.shape, images.dtype)
    print('Label tensor in each batch:', labels.shape, labels.dtype)

    model = resnet(50)
    # print(model)
    FEATS = np.array(test(model, testset_loader, test_set.fileindices, pretrained_path='./save_model/resnet152_d0.5_b32_0.8770.pth', save_path=save_path))