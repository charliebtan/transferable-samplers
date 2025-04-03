import torch
from scipy.spatial.transform import Rotation as R


class Random3DRotationTransform(torch.nn.Module):
    def __init__(self, num_dimensions):
        super().__init__()
        self.num_dimensions = num_dimensions

    def forward(self, data):
        data = data.reshape(1, -1, self.num_dimensions)  # batch dimension needed for einsum
        rot = torch.tensor(R.random(len(data)).as_matrix()).to(data)
        data = torch.einsum("bij,bki->bkj", rot, data)
        data = data.reshape(-1)  # don't want to return with batch dim
        return data
