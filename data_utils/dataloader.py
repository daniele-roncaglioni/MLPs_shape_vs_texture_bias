import os
from pathlib import Path
from typing import List

import torch
import torchvision
from torch.utils.data import default_collate
from torchvision.transforms import transforms, v2

from .data_stats import *


def get_loader(*args, **kwargs):
    try:
        import ffcv
        return get_loader_ffcv(*args, *kwargs)
    except ImportError:
        return get_loader_torch(*args, **kwargs)


def get_loader_torch(
        dataset,
        bs,
        mode,
        augment,
        dev,
        data_resolution=None,
        crop_resolution=None,
        crop_ratio=(0.75, 1.3333333333333333),
        crop_scale=(0.08, 1.0),
        num_samples=None,
        dtype=torch.float32,
        mixup=None,
        data_path='./beton'
):
    mean = MEAN_DICT[dataset]
    std = STD_DICT[dataset]
    if data_resolution is None:
        data_resolution = DEFAULT_RES_DICT[dataset]
    if crop_resolution is None:
        crop_resolution = data_resolution

    transforms_list = [
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.CenterCrop(size=data_resolution),
        transforms.Resize(size=(crop_resolution, crop_resolution), antialias=True),
        transforms.Normalize(mean, std),
    ]

    if augment:
        transforms_list += [
            transforms.RandomResizedCrop((crop_resolution, crop_resolution), scale=crop_scale, ratio=crop_ratio, antialias=True),
            transforms.RandomHorizontalFlip()
        ]
    else:
        transforms_list += [transforms.CenterCrop((crop_resolution, crop_resolution))]

    transforms_pipeline = transforms.Compose(transforms_list)
    if dataset == 'cifar10':
        data = torchvision.datasets.CIFAR10(root=f'{Path(__file__).parent}/data', train=mode == 'train',
                                            download=True, transform=transforms_pipeline)
    elif dataset == 'cifar100':
        data = torchvision.datasets.CIFAR100(root=f'{Path(__file__).parent}/data', train=mode == 'train',
                                             download=True, transform=transforms_pipeline)
    elif dataset == 'imagenette-160':
        data = torchvision.datasets.ImageFolder(root=f'{Path(__file__).parent}/data/imagenette-160/{mode}',
                                                transform=transforms_pipeline)
        if mode == 'test':
            # keep most for validation and small part for real test
            generator = torch.Generator().manual_seed(42)
            [val, test] = torch.utils.data.random_split(data, [0.9, 0.1], generator=generator)
            data = val
    elif dataset == 'imagenette-160-stylized':
        data = torchvision.datasets.ImageFolder(root=f'{Path(__file__).parent}/data/imagenette-160-stylized/{mode}',
                                                transform=transforms_pipeline)
        if mode == 'test':
            # keep most for validation and small part for real test
            generator = torch.Generator().manual_seed(42)
            [val, test] = torch.utils.data.random_split(data, [0.9, 0.1], generator=generator)
            data = val
    else:
        raise ValueError

    if mode == 'train' and augment and mixup > 0:
        def collate_fn(batch):
            return v2.MixUp(alpha=mixup, num_classes=CLASS_DICT[dataset])(*default_collate(batch))
    else:
        collate_fn = None

    return torch.utils.data.DataLoader(data, batch_size=bs, shuffle=True, collate_fn=collate_fn)


# Define an ffcv dataloader
def get_loader_ffcv(
        dataset,
        bs,
        mode,
        augment,
        dev,
        data_resolution=None,
        crop_resolution=None,
        crop_ratio=(0.75, 1.3333333333333333),
        crop_scale=(0.08, 1.0),
        num_samples=None,
        dtype=torch.float32,
        mixup=None,
        data_path='./beton',
):
    from ffcv.fields.decoders import IntDecoder, NDArrayDecoder
    from ffcv.fields.rgb_image import (
        CenterCropRGBImageDecoder,
        RandomResizedCropRGBImageDecoder,
    )
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import (
        Convert,
        ImageMixup,
        LabelMixup,
        RandomHorizontalFlip,
        ToDevice,
        ToTensor,
        ToTorchImage,
    )
    from ffcv.transforms.common import Squeeze

    mode_name = MODE_DICT[dataset] if mode != 'train' else mode
    os_cache = OS_CACHED_DICT[dataset]

    if data_resolution is None:
        data_resolution = DEFAULT_RES_DICT[dataset]
    if crop_resolution is None:
        crop_resolution = data_resolution

    real = '' if dataset != 'imagenet_real' or mode == 'train' else 'real_'
    sub_sampled = '' if num_samples is None or num_samples == SAMPLE_DICT[dataset] else '_ntrain_' + str(num_samples)

    beton_path = os.path.join(
        data_path,
        DATA_DICT[dataset],
        'ffcv',
        mode_name,
        real + f'{mode_name}_{data_resolution}' + sub_sampled + '.beton',
    )

    print(f'Loading {beton_path}')

    mean = MEAN_DICT[dataset]
    std = STD_DICT[dataset]

    if dataset == 'imagenet_real' and mode != 'train':
        label_pipeline: List[Operation] = [NDArrayDecoder()]
    else:
        label_pipeline: List[Operation] = [IntDecoder()]

    if augment:
        image_pipeline: List[Operation] = [
            RandomResizedCropRGBImageDecoder((crop_resolution, crop_resolution), ratio=crop_ratio, scale=crop_scale),
            RandomHorizontalFlip(),
        ]
    else:
        image_pipeline: List[Operation] = [
            CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1)
        ]

    # Add image transforms and normalization
    if mode == 'train' and augment and mixup > 0:
        image_pipeline.extend([ImageMixup(alpha=mixup, same_lambda=True)])
        label_pipeline.extend([LabelMixup(alpha=mixup, same_lambda=True)])

    label_pipeline.extend([ToTensor(), ToDevice(dev, non_blocking=True), Squeeze()])

    image_pipeline.extend(
        [
            ToTensor(),
            ToDevice(dev, non_blocking=True),
            ToTorchImage(),
            Convert(dtype),
            torchvision.transforms.Normalize(mean, std),
        ]
    )

    if mode == 'train':
        num_samples = SAMPLE_DICT[dataset] if num_samples is None else num_samples

        # Shuffle indices in case the classes are ordered
        # indices = list(range(num_samples))

        # random.seed(0)
        # random.shuffle(indices)
        indices = None
    else:
        indices = None

    return Loader(
        beton_path,
        batch_size=bs,
        num_workers=4,
        order=OrderOption.QUASI_RANDOM if mode == 'train' else OrderOption.SEQUENTIAL,
        drop_last=(mode == 'train'),
        pipelines={'image': image_pipeline, 'label': label_pipeline},
        os_cache=os_cache,
        indices=indices,
    )
