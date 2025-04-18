import torch


class PaddingTransform(torch.nn.Module):
    def __init__(self, max_num_particles, num_dimensions):
        super().__init__()
        self.max_num_particles = max_num_particles
        self.num_dimensions = num_dimensions

    def pad_data(self, x):
        assert len(x.shape) == 2
        num_particles = x.shape[0]
        pad_tensor = torch.zeros(self.max_num_particles - num_particles, self.num_dimensions)
        return torch.cat([x, pad_tensor])

    def pad_encoding(self, encoding):
        for key, value in encoding.items():
            encoding[key] = torch.cat([value, torch.zeros(self.max_num_particles - value.shape[0], dtype=torch.int64)])
        return encoding

    def create_mask(self, x):
        num_particles = x.shape[0]
        true_mask = torch.ones(num_particles)
        false_mask = torch.zeros(self.max_num_particles - num_particles)
        return torch.cat([true_mask, false_mask]).bool()

    def forward(self, data):
        x = data["x"]
        encoding = data["encoding"]

        assert len(x.shape) == 2, f"only process single molecules, got shape of {x.shape}"
        assert x.shape[1] == self.num_dimensions, f"expected {self.num_dimensions} dimensions, got {x.shape[1]}"

        x = self.pad_data(x)
        encoding = self.pad_encoding(encoding)
        mask = self.create_mask(x)

        return {
            **data,
            "x": x,
            "encoding": encoding,
            "mask": mask,
        }
