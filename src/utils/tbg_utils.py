import torch


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def kish_effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    """Computes the Kish effective sample size (ESS) for a set of weights.

    Args:
        weights (torch.Tensor): A 1D tensor of sample weights.

    Returns:
        torch.Tensor: The effective sample size (scalar).
    """
    # Sum of weights
    sum_w = torch.sum(weights)
    # Sum of squared weights
    sum_w_sq = torch.sum(weights**2)
    # Kish formula for ESS
    ess = sum_w.pow(2) / sum_w_sq
    return ess
