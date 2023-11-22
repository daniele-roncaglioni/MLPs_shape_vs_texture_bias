from tqdm import tqdm

from data_utils.data_stats import CLASS_DICT
from data_utils.dataloader import get_loader
import torch

from models import get_architecture
from models.networks import get_model
from utils.metrics import AverageMeter, topk_acc, real_acc


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


def predict(dataset, path):
    torch.backends.cuda.matmul.allow_tf32 = True
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"RUNNING ON {device}")

    num_classes = CLASS_DICT[dataset]

    model = get_architecture(architecture='B_6-Wi_512', crop_resolution=64, num_classes=num_classes, device=device).to(device)
    model.load_device = device
    model.load_override(path)

    test_loader = get_loader(dataset, data_resolution=32, crop_resolution=64, bs=16, mode='test', augment=False, dev=device)
    results = test(model, test_loader)
    print(results)


if __name__ == '__main__':

    PATH = '/Users/roncaglionidaniele/Documents/CAS/Deep_Learning/scaling_mlps/checkpoints_finetune/res_64_in21k_cifar10_cifar10/epoch_0'
    predict('cifar10', PATH)
