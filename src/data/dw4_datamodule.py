from typing import Any, Dict, Optional

import numpy as np
import torch
from bgflow import MultiDoubleWellPotential

from src.data.base_datamodule import BaseDataModule
from src.data.components.rotation import Random2DRotationTransform
from src.data.components.transform_dataset import TransformDataset

TRAIN_VAL_TEST_SPLIT = (100_000, 500_000, 500_000)


class DW4DataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = "data/DW4/",
        data_url: str = "https://osf.io/download/a28n7/?view_only=8b2bb152b36f4b6cb8524733623aa5c1",
        filename: str = "dw4-dataidx.npy",
        n_particles: int = 4,
        n_dimensions: int = 2,
        dim: int = 8,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            data_url=data_url,
            filename=filename,
            n_particles=n_particles,
            n_dimensions=n_dimensions,
            dim=dim,
        )
        assert dim == n_particles * n_dimensions

        self.transforms = Random2DRotationTransform(self.n_particles, self.n_dimensions)
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = batch_size
        A = 0.9
        B = -4
        C = 0
        OFFSET = 4
        self.potential = MultiDoubleWellPotential(
            self.dim, self.n_particles, A, B, C, OFFSET, two_event_dims=False
        )

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
        raw_data = np.load(f"{self.hparams.data_dir}{self.hparams.filename}", allow_pickle=True)

        # get the particle data and idx
        data = torch.Tensor(raw_data[0])
        idx = raw_data[1]

        # zero center of mass
        data = self.zero_center_of_mass(data)

        # split the data
        train_data = data[idx[: TRAIN_VAL_TEST_SPLIT[0]]]
        val_data = data[idx[TRAIN_VAL_TEST_SPLIT[0] : TRAIN_VAL_TEST_SPLIT[1]]]
        test_data = data[idx[TRAIN_VAL_TEST_SPLIT[2] :]]

        # compute std on only train data
        self.std = train_data.std()

        # standardize the data
        train_data = self.normalize(train_data)
        val_data = self.normalize(val_data)
        test_data = self.normalize(test_data)

        self.data_train = TransformDataset(train_data, transform=self.transforms)
        self.data_val = val_data
        self.data_test = test_data

    def get_dataset_fig(
        self,
        samples,
        log_p_samples: torch.Tensor,
        samples_jarzynski: torch.Tensor = None,
        jarzynski_log_p: torch.Tensor = None,
        min_energy=-26,
        max_energy=0,
        ylim=(0, 0.2),
        **kwargs,
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
    _ = DW4DataModule()
