import torch
# from fvcore.nn import FlopCountAnalysis


def get_compute(model, dataset_size, res, device):
    input = torch.randn(1, 3 * res * res).to(device)
    flops = FlopCountAnalysis(model, input)

    return flops.total() * 3 * dataset_size
