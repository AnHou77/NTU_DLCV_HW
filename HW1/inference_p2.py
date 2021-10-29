from train_p2 import VGG16_FCN32s, p2, VGG16_FCN8s, test
from torch.utils.data import DataLoader
import sys
import numpy as np


if __name__ == '__main__':
    # load the testset

    data_path = sys.argv[1]
    save_path = sys.argv[2]

    test_set = p2(root=data_path)

    print('# images in testset:', len(test_set))

    # Use the torch dataloader to iterate through the dataset
    testset_loader = DataLoader(test_set, batch_size= 1, shuffle=False)

    # get some random training images
    dataiter = iter(testset_loader)
    images, labels, _ = dataiter.next()

    print('Image tensor in each batch:', images.shape, images.dtype)
    print('Label tensor in each batch:', labels.shape, labels.dtype)

    model = VGG16_FCN8s(7)
    test(model, testset_loader, pretrained_path='./save_model/fcn8_0.7011.pth', save_path='./save_data')
    
    # model = VGG16_FCN32s(7)
    # test(model, testset_loader, pretrained_path='./save_model/fcn32_0.6880.pth', save_path='./save_data')