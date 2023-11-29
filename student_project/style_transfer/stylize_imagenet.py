import sys
import argparse
import os
import shutil
import time
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import save_image, make_grid
# from tensorboardX import SummaryWriter
from PIL import Image

import general as g
import adain
from pathlib import Path

from utils.config import style_info

#parent_dir = os.path.abspath('..\..')
#sys.path.append(parent_dir)
#from parent_dir/utils import get_stylize_parser
from utils.parsers import get_stylize_parser

#####################################################################
# purpose of this file:
# preprocess complete ImageNet (train + val) with AdaIN style
# transfer to speed-up later training.
#####################################################################




def stylize_cifar(args):
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root='../../data_utils/data', train=1,
                                            download=True, transform=transforms.ToTensor())
        foldername = 'cifar10_stylized'
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR10(root=f'{Path(__file__).parent}/data', train=1,
                                            download=True, transform=transforms.ToTensor())
        foldername = 'cifar100_stylized'
    else:
        raise ValueError('dataset must be either cifar10 or cifar100')

    mode = args.mode

    trg_path = f'{Path(__file__).parent.parent}/data/{foldername}/{mode}/'
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    style_transfer = get_style_loader()

    process_TorchDataloader(dataloader, trg_path, input_transforms=[style_transfer])





def get_style_loader():
    #############################################################
    #         START STYLE TRANSFER SETUP
    #############################################################

    style_dir = g.ADAIN_PREPROCESSED_PAINTINGS_DIR
    assert len(os.listdir(style_dir)) == 79395

    do_style_preprocessing = False
    num_styles = len(os.listdir(style_dir))
    print("=> Using " + str(num_styles) + " different style images.")
    all_styles = [[] for _ in range(num_styles)]
    for i, name in enumerate(sorted(os.listdir(style_dir))):
        all_styles[i] = os.path.join(style_dir, name)

    transfer_args = g.get_default_adain_args()
    transferer = adain.AdaIN(transfer_args)
    print("=> Succesfully loaded style transfer algorithm.")

    style_loader = adain.StyleLoader(style_transferer=transferer,
                                     style_img_file_list=all_styles,
                                     rng=np.random.RandomState(seed=49809),
                                     do_preprocessing=do_style_preprocessing)

    style_transfer = style_loader.get_style_tensor_function
    print("=> Succesfully created style loader.")
    return style_transfer


def main(args):
    assert (args.mode=='train' or args.mode=='val'), "the mode must be either train or val"
    # Data loading code
    #traindir = os.path.join(args.dataset_source_path, 'train')
    #valdir = os.path.join(args.dataset_source_path, 'val')

    source_dir = os.path.join(args.dataset_source_path, f'{args.mode}/')
    target_dir = os.path.join(args.dataset_target_path, f"{args.mode}/")

    style_transfer = get_style_loader()

    #############################################################
    #         CREATING DATA LOADERS
    #############################################################

    class MyDataLoader():
        """Convenient data loading class."""

        def __init__(self, root,
                     transform=transforms.ToTensor(),
                     target_transform=None,
                     batch_size=args.batch_size,
                     num_workers=args.workers,
                     shuffle=False,
                     sampler=None):
            self.dataset = datasets.ImageFolder(root=root,
                                                transform=transform,
                                                target_transform=target_transform)
            self.loader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True)
    transform_list = [transforms.Resize(256)(x)] if args.resize else []
    transform_list.extend([
        transforms.CenterCrop(args.imgsize_target),
        transforms.ToTensor()])
    default_transforms = transforms.Compose(transform_list)


    loader = MyDataLoader(root=source_dir,
                                    transform=default_transforms,
                                    shuffle=False,
                                    sampler=None)

    print("=> Succesfully created all data loaders.")
    print("")

    #############################################################
    #         PREPROCESS DATASETS
    #############################################################

    print("Preprocessing training data:")
    preprocess(data_loader=loader,
               input_transforms=[style_transfer],
               sourcedir=source_dir,
               targetdir=target_dir)

def process_TorchDataloader(data_loader, targetdir, input_transforms=None):
    cnter = 0
    for i, (input, target) in enumerate(data_loader):
        # apply manipulations
        for transform in input_transforms:
            input = transform(input)
        for img_index in range(input.size()[0]):
            trg_classdir = os.path.join(targetdir, str(target[img_index].item()))

            if not os.path.exists(trg_classdir):
                os.makedirs(trg_classdir)
            target_img_path = os.path.join(trg_classdir,
                                           f'{target[img_index].item()}_{cnter}.png')

            save_image(tensor=input[img_index, :, :, :],
                       fp=target_img_path)

            cnter += 1

def preprocess(data_loader, sourcedir, targetdir,
               input_transforms=None):
    """Preprocess ImageNet with certain transformations.

    Keyword arguments:
    sourcedir -- a directory path, e.g. /bla/imagenet/train/
                 where subdirectories correspond to single classes
                 (need to be filled with .JPEG images)
    targetdir -- a directory path, e.g. /bla/imagenet-new/train/
                 where sourcedir will be mirrored, except
                 that images will be preprocessed and saved
                 as .png instead of .JPEG
    input_transforms -- a list of transformations that will
                        be applied (e.g. style transfer)
    """

    counter = 0
    current_class = None
    current_class_files = None

    # create list of all classes
    #    Edited by Tristan: get classes always from train set as val and test are all in one big folder
    all_classes = sorted(os.listdir(sourcedir))
    all_classes = [s for s in all_classes if s.__contains__('n')]

    for i, (input, target) in enumerate(data_loader.loader):

        # apply manipulations
        for transform in input_transforms:
            input = transform(input)

        for img_index in range(input.size()[0]):

            # for each image in a batch:
            # - determine ground truth class
            # - transform image
            # - save transformed image in new directory
            #   with the same class name

            # the mapping between old and new filenames
            # is achieved by looking at the indices of
            # the sorted(os.listdir()) results.

            source_class = all_classes[target[img_index]]
            source_classdir = os.path.join(sourcedir, source_class)
            if not os.path.exists(source_classdir):
                os.makedirs(source_classdir)

            assert os.path.exists(source_classdir)

            target_classdir = os.path.join(targetdir, source_class)
            if not os.path.exists(target_classdir):
                os.makedirs(target_classdir)

            if source_class != current_class:
                # moving on to new class:
                # start counter (=index) by 0, update list of files
                # for this new class
                counter = 0
                current_class_files = sorted(os.listdir(source_classdir))

            current_class = source_class

            target_img_path = os.path.join(target_classdir,
                                           current_class_files[counter].replace(".JPEG", ".png"))

            save_image(tensor=input[img_index, :, :, :],
                       fp=target_img_path)
            counter += 1

        if i % args.print_freq == 0:
            print('Progress: [{0}/{1}]\t'
                .format(
                i, len(data_loader.loader)))


if __name__ == '__main__':
    parser = get_stylize_parser()
    args = parser.parse_args()

    dataset_source_path, dataset_target_path, imgsize_target, resize = style_info()
    d = vars(args)
    d["dataset_source_path"] = dataset_source_path[args.dataset]
    d["dataset_target_path"] = dataset_target_path[args.dataset]
    d["imgsize_target"] = imgsize_target[args.dataset]
    d['resize'] = resize[args.dataset] # resize image to 256 before cropping as in the original implementation?
    d['mode'] = 'val'  # 'val'


    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        stylize_cifar(args)
    else:
        main(args)