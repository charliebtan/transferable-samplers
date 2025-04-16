import torch
from scipy.spatial.transform import Rotation as R


class Random3DRotationTransform(torch.nn.Module):
    def __init__(self, num_dimensions):
        super().__init__()
        self.num_dimensions = num_dimensions

    def forward(self, data):
        x = data["x"]
        assert len(x.shape) == 1, f"only process single molecules, got shape of {x.shape}"

        mask = data.get("mask", None)
        num_particles = mask.sum() if mask is not None else x.shape[0]

        x = x.reshape(1, -1, self.num_dimensions)  # batch dimension needed for einsum
        x, padding = x[:, :num_particles], x[:, num_particles:]  # slice out the data and padding

        assert torch.sum(padding) == 0, "padding should be zero"

        rot = torch.tensor(R.random(len(x)).as_matrix()).to(x)
        x = torch.einsum("bij,bki->bkj", rot, x)

        x = torch.cat([x, padding], dim=1)  # re-add the padding back

        # Reshape back to original shape
        x = x.reshape(-1)

        data.update({"x": x})

        return data
