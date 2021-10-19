from train import p1, resnet, test
from torch.utils.data import DataLoader
import sys

if __name__ == '__main__':
    # load the testset

    data_path = sys.argv[1]
    save_path = sys.argv[2]

    test_set = p1(root=data_path, target= 'test')

    print('# images in testset:', len(test_set))

    # Use the torch dataloader to iterate through the dataset
    testset_loader = DataLoader(test_set, batch_size= 50, shuffle=False, num_workers=4)

    # get some random training images
    dataiter = iter(testset_loader)
    images, labels = dataiter.next()

    print('Image tensor in each batch:', images.shape, images.dtype)
    print('Label tensor in each batch:', labels.shape, labels.dtype)

    model = resnet(50)

    # print(model)
    test(model, testset_loader, test_set.fileindices, pretrained_path='./save_model/resnet152_d0.5_b32_0.8770.pth', save_path=save_path)