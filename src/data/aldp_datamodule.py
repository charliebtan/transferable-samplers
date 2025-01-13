from typing import Any, Dict, Optional

import numpy as np
import torch
from bgmol.datasets import AImplicitUnconstrained

from src.data.base_datamodule import BaseDataModule
from src.data.components.rotation import Random3DRotationTransform
from src.data.components.transform_dataset import TransformDataset


class ALDPDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = "data/AD2/",
        data_url: str = "https://osf.io/download/y7ntk/?view_only=1052300a21bd43c08f700016728aa96e",
        filename: str = "AD2_weighted.npy",
        n_particles: int = 22,
        n_dimensions: int = 3,
        dim: int = 66,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        scaling: float = 10.0,
    ) -> None:
        super().__init__()
        assert dim == n_particles * n_dimensions

        self.transforms = Random3DRotationTransform()
        self.scaling = scaling

        self.batch_size_per_device = batch_size
        self.bgmol_dataset = AImplicitUnconstrained(read=True, download=False)
        self.potential = self.bgmol_dataset.get_energy_model()

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

        # load the data + tensorize
        train_data = np.load(f"{self.hparams.data_dir}/{self.hparams.filename}", allow_pickle=True)

        # data is 10 times larger in bgflow dataset than in numpy
        train_data = torch.tensor(train_data) / self.scaling

        # compute std on only train data
        train_data = train_data.reshape(-1, self.n_particles, self.n_dimensions)
        self.std = train_data.std()

        # standardize the data
        train_data = self.standardize(train_data)
        test_data = self.standardize(self.dataset.xyz)

        # split the data
        self.data_train = TransformDataset(train_data, transform=self.transforms)
        self.data_val = test_data[:100000]  # TODO hardcode
        self.data_test = test_data[100000:200000]

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


if __name__ == "__main__":
    _ = ALDPDataModule()
