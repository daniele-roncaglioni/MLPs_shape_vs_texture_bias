import torch
import numpy
import torchvision.datasets as datasets
from torchvision import transforms

from data_utils.data_stats import *
from utils.parsers import *

#cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

def calc_stats(dataset, args):
    imgs = [item[0] for item in dataset] # item[0] and item[1] are image and its label
    imgs = torch.stack([transforms.CenterCrop(160)(transforms.ToTensor()(img)) for img in imgs], dim=0).numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()
    print(mean_r,mean_g,mean_b)

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    print(std_r,std_g,std_b)

if __name__ == '__main__':
    parser = get_training_parser()
    args = parser.parse_args()

    args.num_classes = CLASS_DICT[args.dataset]

    if args.n_train is None:
        args.n_train = SAMPLE_DICT[args.dataset]

    if args.crop_resolution is None:
        args.crop_resolution = args.resolution

    mode = 'train'
    dataset = datasets.ImageFolder(root=f'../../data_utils/data/imagenette-160/{mode}')
    calc_stats(dataset, args)
