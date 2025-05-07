"""
Temp workaround. Need this to load pre-trained models since the loading expects
SMCSampler to be in the src.model.components dir
"""

import torch


class SMCSampler(torch.nn.Module):
    pass
