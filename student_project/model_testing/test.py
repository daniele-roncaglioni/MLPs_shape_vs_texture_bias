from tqdm import tqdm

from data_utils.data_stats import CLASS_DICT
from data_utils.dataloader import get_loader
import torch

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
    num_classes = CLASS_DICT[dataset]
    model = get_model(architecture='B_6-Wi_1024', resolution=64, num_classes=num_classes, checkpoint=f'in21k_{dataset}', device=device).to(device)
    test_loader = get_loader(dataset, bs=16, mode='test', augment=False, dev='cpu')
    results = test(model, test_loader, dataset)
    print(results)


if __name__ == '__main__':
    predict('cifar10')
