"""Adapted from https://github.com/jarridrb/DEM/blob/main/dem/models/components/prioritised_replay_buffer.py"""

from typing import Callable, NamedTuple

import numpy as np
import torch


class ReplayData(NamedTuple):
    """Samples generated from model or sampling alg."""

    x: torch.Tensor
    seq_name: list[str]


def sample_without_replacement(logits: torch.Tensor, n: int) -> torch.Tensor:
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    z = torch.distributions.Gumbel(torch.tensor(0.0), torch.tensor(1.0)).sample(logits.shape).to(logits.device)
    topk = torch.topk(z + logits, n, sorted=False)
    indices = topk.indices
    indices = indices[torch.randperm(n).to(indices.device)]
    return indices


class ReplayBuffer:
    def __init__(
        self,
        dim: int,
        max_length: int,
        min_sample_length: int,
        initial_sampler: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        device: str = "cpu",
        sample_with_replacement: bool = False,
        fill_buffer_during_init: bool = True,
    ):
        """
        Create prioritised replay buffer for batched sampling and adding of data.
        Args:
            dim: dimension of x data
            max_length: maximum length of the buffer
            min_sample_length: minimum length of buffer required for sampling
            initial_sampler: sampler producing x, log_w and log q, used to fill the buffer up to
                the min sample length. The initialised flow + AIS may be used here,
                or we may desire to use AIS with more distributions to give the flow a "good start".
            device: replay buffer device
            sample_with_replacement: Whether to sample from the buffer with replacement.
            fill_buffer_during_init: Whether to use `initial_sampler` to fill the buffer initially.
                If a checkpoint is going to be loaded then this should be set to False.

        The `max_length` and `min_sample_length` should be sufficiently long to prevent overfitting
        to the replay data. For example, if `min_sample_length` is equal to the
        sampling batch size, then we may overfit to the first batch of data, as we would update
        on it many times during the start of training.
        """
        assert min_sample_length < max_length
        self.dim = dim
        self.max_length = max_length
        self.min_sample_length = min_sample_length
        self.buffer = torch.empty(size=(0, dim))
        self.seq_names = []

        self.possible_indices = torch.arange(self.max_length).to(device)
        self.device = device
        self.sample_with_replacement = sample_with_replacement

        if fill_buffer_during_init:
            while self.can_sample is False:
                # fill buffer up minimum length
                x = initial_sampler()
                self.add(x)
        else:
            print("Buffer not initialised, expected that checkpoint will be loaded.")

    def __len__(self):
        return len(self.buffer)

    @torch.no_grad()
    def add(self, x: torch.Tensor, seq_name: str = None) -> None:
        """Add a new batch of generated data to the replay buffer."""
        x = x.detach().cpu()
        self.buffer = torch.concat([self.buffer, x], dim=0)
        self.seq_names = self.seq_names + [seq_name] * len(x)
        self.buffer = self.buffer[-self.max_length :]
        self.seq_names = self.seq_names[-self.max_length :]

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a batch of sampled data, if the batch size is specified then the batch will have
        a leading axis of length batch_size, otherwise the default self.batch_size will be used."""

        max_index = min(self.max_length, len(self.buffer))
        if self.sample_with_replacement:
            indices = torch.randint(max_index, (batch_size,))
        else:
            indices = torch.randperm(max_index)[:batch_size]

        x, seq_name, indices = (
            self.buffer[indices],
            np.array(self.seq_names)[indices],
            indices,
        )
        return x, seq_name, indices

    def save(self, path):
        """Save buffer to file."""
        to_save = {
            "x": self.buffer.x.detach().cpu(),
            "current_index": self.current_index,
            "is_full": self.is_full,
            "can_sample": self.can_sample,
        }
        torch.save(to_save, path)

    def load(self, path):
        """Load buffer from file."""
        old_buffer = torch.load(path)
        indices = torch.arange(self.max_length)
        self.buffer.x[indices] = old_buffer["x"].to(self.device)
        self.current_index = old_buffer["current_index"]
        self.is_full = old_buffer["is_full"]
        self.can_sample = old_buffer["can_sample"]


# if __name__ == "__main__":
#     # to check that the replay buffer runs
#     dim = 3 * 22
#     batch_size = 3
#     n_batches_total_length = 2
#     # length = n_batches_total_length * batch_size
#     length = 100
#     min_sample_length = int(length * 0.5)

#     def initial_sampler():
#         return torch.ones(batch_size, dim)

#     buffer = ReplayBuffer(dim, length, min_sample_length, initial_sampler)
#     n_batches = 3
#     for i in range(100):
#         buffer.add(torch.ones(batch_size, dim))
#         x, indices = buffer.sample(batch_size)

#     x, indices = buffer.sample(batch_size=10)
#     print(f"x shape: {x.shape}")
#     print(x[:4])
#     dataset = buffer.sample_n_batches(batch_size=10, n_batches=10)


# class ReplayBuffer:
#     def __init__(
#         self,
#         dim: int,
#         max_length: int,
#         min_sample_length: int = 0,
#         initial_sampler: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
#         device: str = "cpu",
#         sample_with_replacement: bool = False,
#         fill_buffer_during_init: bool = True,
#     ):
#         """
#         Create prioritised replay buffer for batched sampling and adding of data.
#         Args:
#             dim: dimension of x data
#             max_length: maximum length of the buffer
#             min_sample_length: minimum length of buffer required for sampling
#             initial_sampler: sampler producing x, log_w and log q, used to fill the buffer up to
#                 the min sample length. The initialised flow + AIS may be used here,
#                 or we may desire to use AIS with more distributions to give the flow a "good start".
#             device: replay buffer device
#             sample_with_replacement: Whether to sample from the buffer with replacement.
#             fill_buffer_during_init: Whether to use `initial_sampler` to fill the buffer initially.
#                 If a checkpoint is going to be loaded then this should be set to False.

#         The `max_length` and `min_sample_length` should be sufficiently long to prevent overfitting
#         to the replay data. For example, if `min_sample_length` is equal to the
#         sampling batch size, then we may overfit to the first batch of data, as we would update
#         on it many times during the start of training.
#         """
#         assert min_sample_length < max_length
#         self.dim = dim
#         self.max_length = max_length
#         self.min_sample_length = min_sample_length
#         self.buffer = ReplayData(
#             x=torch.zeros(self.max_length, dim).to(device),
#             dim=torch.zeros(
#                 self.max_length,
#             ).to(device),
#         )
#         self.possible_indices = torch.arange(self.max_length).to(device)
#         self.device = device
#         self.current_index = 0
#         self.is_full = False  # whether the buffer is full
#         self.sample_with_replacement = sample_with_replacement

#     def __len__(self):
#         if self.is_full:
#             return self.max_length
#         else:
#             return self.current_index

#     @torch.no_grad()
#     def add(self, x: torch.Tensor) -> None:
#         """Add a new batch of generated data to the replay buffer."""
#         batch_size = x.shape[0]
#         x = x.to(self.device)
#         indices = (torch.arange(batch_size) + self.current_index).to(self.device) % self.max_length
#         self.buffer.x[indices, : x.shape[-1]] = x
#         self.buffer.dim[indices] = x.shape[-1]
#         new_index = self.current_index + batch_size
#         if not self.is_full:
#             self.is_full = new_index >= self.max_length
#             self.can_sample = new_index >= self.min_sample_length
#         self.current_index = new_index % self.max_length

#     def get_last_n_inserted(self, num_to_get: int) -> torch.Tensor:
#         if self.is_full:
#             assert num_to_get <= self.max_length
#         else:
#             assert num_to_get < self.current_index

#         start_idx = self.current_index - num_to_get
#         idxs = [torch.arange(max(start_idx, 0), self.current_index)]
#         if start_idx < 0:
#             idxs.append(torch.arange(self.max_length + start_idx, self.max_length))

#         idx = torch.cat(idxs)

#         return self.buffer.x[idx]

#     @torch.no_grad()
#     def sample(
#         self,
#         batch_size: int,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """Return a batch of sampled data, if the batch size is specified then the batch will have
#         a leading axis of length batch_size, otherwise the default self.batch_size will be used."""
#         # if not self.can_sample:
#         #     raise Exception("Buffer must be at minimum length before calling sample")

#         max_index = self.max_length if self.is_full else self.current_index
#         if self.sample_with_replacement:
#             indices = torch.randint(max_index, (batch_size,)).to(self.device)
#         else:
#             indices = torch.randperm(max_index)[:batch_size].to(self.device)

#         dim = self.buffer.dim[indices]
#         x, indices = (
#             self.buffer.x[indices, :dim],
#             indices,
#         )
#         return x, indices

#     def sample_n_batches(self, batch_size: int, n_batches: int) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
#         """Returns a list of batches."""
#         x, indices = self.sample(batch_size * n_batches)
#         x_batches = torch.chunk(x, n_batches)
#         indices_batches = torch.chunk(indices, n_batches)
#         dataset = [(x, indxs) for x, indxs in zip(x_batches, indices_batches)]
#         return dataset

#     def save(self, path):
#         """Save buffer to file."""
#         to_save = {
#             "x": self.buffer.x.detach().cpu(),
#             "current_index": self.current_index,
#             "is_full": self.is_full,
#             "can_sample": self.can_sample,
#         }
#         torch.save(to_save, path)

#     def load(self, path):
#         """Load buffer from file."""
#         old_buffer = torch.load(path)
#         indices = torch.arange(self.max_length)
#         self.buffer.x[indices] = old_buffer["x"].to(self.device)
#         self.current_index = old_buffer["current_index"]
#         self.is_full = old_buffer["is_full"]
#         self.can_sample = old_buffer["can_sample"]


# if __name__ == "__main__":
#     # to check that the replay buffer runs
#     dim = 3 * 22
#     batch_size = 3
#     n_batches_total_length = 2
#     # length = n_batches_total_length * batch_size
#     length = 100
#     min_sample_length = int(length * 0.5)

#     def initial_sampler():
#         return torch.ones(batch_size, dim)

#     buffer = ReplayBuffer(dim, length, min_sample_length, initial_sampler)
#     n_batches = 3
#     for i in range(100):
#         buffer.add(torch.ones(batch_size, dim))
#         x, indices = buffer.sample(batch_size)

#     x, indices = buffer.sample(batch_size=10)
#     print(f"x shape: {x.shape}")
#     print(x[:4])
#     dataset = buffer.sample_n_batches(batch_size=10, n_batches=10)
