import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from tqdm import tqdm

from data_utils.data_stats import CLASS_DICT, STD_DICT, MEAN_DICT
from data_utils.dataloader import get_loader
import torch

from models import get_architecture
from utils.metrics import AverageMeter, topk_acc


def imshow(img, dataset):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=1. / STD_DICT[dataset]),
                                   transforms.Normalize(mean=-MEAN_DICT[dataset],
                                                        std=[1., 1., 1.]),
                                   ])

    inv_tensor = invTrans(img)
    npimg = inv_tensor.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


@torch.no_grad()
def plot_batch_predictions(model, loader, dataset, classes):
    ims, targs = next(iter(loader))
    ims_flat = torch.reshape(ims, (ims.shape[0], -1))
    preds = model(ims_flat)
    imshow(torchvision.utils.make_grid(ims), dataset)
    # print labels
    print('Ground truth: ', ' '.join(f'{classes[targ.item()]:5s}' for targ in targs))
    print('Predicted   : ', ' '.join(f'{classes[torch.argmax(pred).item()]:5s}' for pred in preds))


@torch.no_grad()
def test(model, loader):
    model.eval()
    total_acc, total_top5 = AverageMeter(), AverageMeter()

    for ims, targs in tqdm(loader, desc="Evaluation"):
        ims = torch.reshape(ims, (ims.shape[0], -1))
        preds = model(ims)

        acc, top5 = topk_acc(preds, targs, k=5, avg=True)

        total_acc.update(acc, ims.shape[0])
        total_top5.update(top5, ims.shape[0])

    return (
        total_acc.get_avg(percentage=True),
        total_top5.get_avg(percentage=True),
    )


def get_test_artifacts(model_config, dataset, bs):
    torch.backends.cuda.matmul.allow_tf32 = True
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"RUNNING ON {device}")

    num_classes = CLASS_DICT[dataset]

    model = get_architecture(architecture=model_config['architecture'], crop_resolution=model_config['net_input_res'], num_classes=num_classes, device=device).to(device)
    model.load_device = device
    model.load_override(model_config['checkpoint_path'])

    test_loader = get_loader(dataset, crop_resolution=model_config['net_input_res'], bs=bs, mode='test', augment=False, dev=device)
    return model, test_loader


if __name__ == '__main__':
    model_config = {
        'architecture': 'B_6-Wi_512',
        'net_input_res': 64,
        'checkpoint_path': '/Users/roncaglionidaniele/Documents/CAS/Deep_Learning/scaling_mlps/checkpoints_finetune/res_64_in21k_cifar10_cifar10/epoch_0'
    }
    dataset = 'cifar10'
    batch_size = 16

    model, test_loader = get_test_artifacts(model_config, dataset, batch_size)

    results = test(model, test_loader)
    print(results)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    plot_batch_predictions(model, test_loader, dataset, classes)
