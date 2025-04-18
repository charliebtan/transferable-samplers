import torch


class AddEncodingTransform(torch.nn.Module):
    def __init__(self, encoding_dict):
        super().__init__()
        self.encoding_dict = encoding_dict

    def forward(self, data):
        seq_name = data["seq_name"]
        return {
            **data,
            "encoding": self.encoding_dict[seq_name],
        }
