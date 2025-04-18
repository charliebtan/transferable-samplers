import torch
from scipy.spatial.transform import Rotation as R


class Random3DRotationTransform(torch.nn.Module):
    def __init__(self, num_dimensions):
        super().__init__()
        self.num_dimensions = num_dimensions

    def forward(self, data):
        x = data["x"]

        assert len(x.shape) == 2, f"only process single molecules, got shape of {x.shape}"
        assert x.shape[1] == self.num_dimensions, f"expected {self.num_dimensions} dimensions, got {x.shape[1]}"

        x = x.unsqueeze(0)

        rot = torch.tensor(R.random(len(x)).as_matrix()).to(x)
        x = torch.einsum("bij,bki->bkj", rot, x)

        x = x.squeeze(0)

        return {
            **data,
            "x": x,
        }
