from torch import topk, any, sum
import torch


# from prettytable import PrettyTable


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_avg(self, percentage=False):
        return self.sum / self.count if not percentage else self.sum * 100 / self.count


def topk_acc(preds, targs, targs_perm=None, k=5, avg=False, mixup=True):
    if avg:
        num = preds.shape[0]
    else:
        num = 1
    _, top_k_inds = topk(preds, k)
    if mixup:
        top_5 = torch.tensor([0])  # we don't care about top 5 because we have 10 classes only in our dataset
        acc = 1 / num * sum(top_k_inds[:, 0].eq(torch.argmax(targs, dim=1)), dim=0)
    else:
        top_5 = 1 / num * sum(any(top_k_inds == targs.unsqueeze(dim=1), dim=1), dim=0)
        acc = 1 / num * sum(top_k_inds[:, 0].eq(targs), dim=0)

    if targs_perm is not None:
        top_5_perm = (
                1 / num * sum(any(top_k_inds == targs_perm.unsqueeze(dim=1), dim=1), dim=0)
        )
        acc_perm = 1 / num * sum(top_k_inds[:, 0].eq(targs_perm), dim=0)

        return torch.maximum(acc, acc_perm), torch.maximum(top_5, top_5_perm)

    return acc.item(), top_5.item()


def real_acc(preds, targs, k, avg=False):
    if avg:
        num = preds.shape[0]
    else:
        num = 1
    _, top_k_inds = topk(preds, k)
    top_1_inds = top_k_inds[:, 0]
    acc_real = 1 / num * sum(any(top_1_inds.unsqueeze(dim=1).eq(targs), dim=1), dim=0)

    return acc_real

# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params += params
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params
