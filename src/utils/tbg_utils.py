import torch


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def effective_sample_size(log_weights):
    """Kish effective sample size; log weights don't have to be normalized"""
    return torch.exp(
        2 * torch.logsumexp(log_weights, dim=0) - torch.logsumexp(2 * log_weights, dim=0)
    )


def sampling_efficiency(log_weights):
    """Kish effective sample size / sample size; log weights don't have to be normalized"""
    return effective_sample_size(log_weights) / len(log_weights)
