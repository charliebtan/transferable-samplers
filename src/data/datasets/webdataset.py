import webdataset as wds
import torch
import glob
import os
import numpy as np

def build_webdataset(
    path: str,
    tar_pattern: str,
    num_dimensions: int = 3,
    shuffle_buffer: int = 1000,
    max_seq_len: int = None,
    transform=None,
):
    TAR_DIRS = ["/scratch/t/tanc/webdataset_4", "/project/aip-necludov/tanc/webdataset_4"]

    # === Gather .tar files ===
    tar_paths = []
    for dir_path in TAR_DIRS:
        tar_paths.extend(glob.glob(os.path.join(dir_path, "*.tar")))
    assert tar_paths, "No .tar files found!"

    # def log_shard(sample):
    #     shard = sample.get("__url__", "unknown")
    #     print(f"Rank {torch.distributed.get_rank()} reading from: {shard} (PID {os.getpid()})")
    #     return sample

    def make_sample(sample):
        key, x = sample
        sequence = key.split("_")[0]
        x = np.frombuffer(x, dtype=np.float32).reshape(-1, num_dimensions)
        x = torch.from_numpy(x.copy())
        sample_dict = {"x": x, "sequence": sequence}
        if transform:
            sample_dict = transform(sample_dict)
            sample_dict["x"] = sample_dict["x"].view(-1)
        return sample_dict

    dataset = (
        wds.WebDataset(
            tar_paths,
            shardshuffle=False, # Ignored when `resampled=True`
            resampled=True,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )
        .shuffle(shuffle_buffer)
        # .map(log_shard)
        .to_tuple("__key__", "bin")
        .select(
            lambda sample: len(sample[0].split("_")[0]) <= max_seq_len
            if max_seq_len is not None
            else True
        )
        .map(make_sample)
    )

    return dataset
