from typing import Any, Dict, Optional

import numpy as np
import torch
from bgmol.datasets import AImplicitUnconstrained
from src.data.components.rotation import create_random_rotation_matrix
from torch.utils.data import DataLoader, Dataset

from .dw4_datamodule import DW4DataModule


class AldpDataModule(DW4DataModule):
    def __init__(
        self,
        data_dir: str = "data/AD2/",
        filename: str = "AD2_weighted.npy",
        n_particles: int = 22,
        n_dimensions: int = 3,
        dim: int = 66,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super(DW4DataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.dim = dim
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.dataset = AImplicitUnconstrained(read=True, download=False)
        self.potential = self.dataset.get_energy_model()

        self.batch_size_per_device = batch_size
        self.curr_loads = 0
        self.curr_epoch = 0

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # TODO bit odd to do at every setup call? probably not too slow...

        # load the data + tensorize
        data = np.load(f"{self.hparams.data_dir}/{self.hparams.filename}", allow_pickle=True)
        self.scaling = 10
        # data is 10 times larger in bgflow dataset than in numpy
        all_data = torch.tensor(data) / self.scaling
        all_data = all_data.reshape(-1, self.n_particles, self.n_dimensions)
        self.std = all_data.std()
        print(f"Standard deviation: {self.std}")
        # standardize the data
        all_data = all_data - all_data.mean(axis=1, keepdims=True)
        all_data = all_data / self.std

        test_data = self.dataset.xyz
        test_data = test_data - test_data.mean(axis=1, keepdims=True)
        test_data = test_data / self.std
        test_data = test_data.reshape(-1, self.dim)

        # rotation augementation
        rot = create_random_rotation_matrix(len(all_data))
        all_data = torch.einsum("bij,bki->bkj", rot, all_data)

        # return to vector shape
        all_data = all_data.reshape(-1, self.dim)

        # split the data
        self.data_train = all_data
        self.data_val = test_data[:100000]
        self.data_test = test_data[100000:200000]
        self.curr_loads += 1
        print(f"XXXXXXXXXX Data loaded {self.curr_loads} times XXXXXXXXXXXX")

    def get_dataset_fig(
        self,
        samples,
        log_p_samples: torch.Tensor,
        samples_jarzynski: torch.Tensor = None,
        jarzynski_log_p: torch.Tensor = None,
        min_energy=-50,
        max_energy=100,
        ylim=(0, 0.2),
    ):
        return super().get_dataset_fig(
            samples,
            log_p_samples,
            samples_jarzynski,
            jarzynski_log_p,
            min_energy,
            max_energy,
            ylim=ylim,
        )

    def unnormalize(self, x):
        assert x.shape[-1] == self.dim
        x = x.reshape(-1, self.n_particles, self.n_dimensions)
        x = x * self.std.to(x)
        x = x.reshape(-1, self.dim)
        return x

    def prepare_data(self) -> None:
        """Prepare data. This method is called by Lightning once when initializing the `DataModule`."""
        pass
