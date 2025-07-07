import webdataset as wds
import torch

def build_webdataset(
    path: str,
    tar_pattern: str,
    num_dimensions: int = 3,
    shuffle_buffer: int = 1000,
    transform=None,
):
    def make_sample(sample):
        key, x = sample
        seq_name = key.split("_")[0]
        x = torch.from_numpy(x).float().view(-1, num_dimensions)
        sample_dict = {"x": x, "seq_name": seq_name}
        if transform:
            sample_dict = transform(sample_dict)
            sample_dict["x"] = sample_dict["x"].view(-1)
        return sample_dict

    dataset = (
        wds.WebDataset(
            f"{path}/{tar_pattern}",
            shardshuffle=False, # Ignored when `resampled=True`
            resampled=True,
            nodesplitter=wds.split_by_node,
        )
        .shuffle(shuffle_buffer)
        .decode()
        .to_tuple("__key__", "npy")
        .map(make_sample)
    )

    return dataset
