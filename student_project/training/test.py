import json
import time

from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor, ConvertImageDtype, InterpolationMode
from tqdm import tqdm

from data_utils.data_stats import MEAN_DICT, STD_DICT, CLASS_DICT
from models import get_architecture
import torch
import torchvision
import torchvision.transforms as transforms

from models.networks import get_model
from utils.metrics import AverageMeter, topk_acc, real_acc


@torch.no_grad()
def test(model, loader, dataset):
    model.eval()
    total_acc, total_top5 = AverageMeter(), AverageMeter()

    for ims, targs in tqdm(loader, desc="Evaluation"):
        ims = torch.reshape(ims, (ims.shape[0], -1))
        preds = model(ims)

        if dataset != 'imagenet_real':
            acc, top5 = topk_acc(preds, targs, k=5, avg=True)
        else:
            acc = real_acc(preds, targs, k=5, avg=True)
            top5 = 0

        total_acc.update(acc, ims.shape[0])
        total_top5.update(top5, ims.shape[0])

    return (
        total_acc.get_avg(percentage=True),
        total_top5.get_avg(percentage=True),
    )


def predict(dataset, device='cpu'):
    # get test data
    mean = MEAN_DICT[dataset]
    std = STD_DICT[dataset]
    transform = transforms.Compose(
        [
            ToTensor(),
            # ToTorchImage(),
            ConvertImageDtype(torch.float32),
            torchvision.transforms.Resize(size=(64, 64), interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.Normalize(mean, std),
        ]
    )
    batch_size = 4
    num_classes = CLASS_DICT[dataset]
    if dataset == 'cifar10':
        test_set = torchvision.datasets.CIFAR10(root='./../data', train=False,
                                                download=True, transform=transform)
    elif dataset == 'cifar100':
        test_set = torchvision.datasets.CIFAR100(root='./../data', train=False,
                                                 download=True, transform=transform)
    else:
        raise ValueError

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    model = get_model(architecture='B_6-Wi_1024', resolution=64, num_classes=num_classes, checkpoint=f'in21k_{dataset}').to(device)

    results = test(model, test_loader, dataset)
    print(results)


if __name__ == '__main__':
    predict('cifar10')
