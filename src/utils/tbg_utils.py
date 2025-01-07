import torch

def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x