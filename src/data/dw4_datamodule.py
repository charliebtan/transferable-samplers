import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

# I've just made these constants, I think it's safer as they should be static
# and it allows me to access them from other files (e.g to destandardize)

DIM = 2
NUM_PARTICLES = 4

TRAIN_VAL_TEST_SPLIT = (100_000, 500_000, 500_000)

# With centering the train split has per-dim stds of:
DW4_STD = [1.8230, 1.8103]


class DW4DataModule(LightningDataModule):
    """TODO (dw4): Add a description.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        data_url: str = "https://osf.io/download/a28n7/?view_only=8b2bb152b36f4b6cb8524733623aa5c1",
        filename="dw4-dataidx.npy",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `DW4DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        os.makedirs(self.hparams.data_dir, exist_ok=True)

        if not os.path.exists(self.hparams.data_dir + self.hparams.filename):
            print(f"Downloading file from {self.hparams.data_url}")
            response = requests.get(self.hparams.data_url, timeout=300)

            # Save the file in binary (write) mode
            with open(self.hparams.data_dir + self.hparams.filename, "wb") as f:
                f.write(response.content)

            print(f"File downloaded and saved as: {self.hparams.filename}")

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
        dw4_data = np.load(f"{self.hparams.data_dir}{self.hparams.filename}", allow_pickle=True)
        all_data = torch.Tensor(dw4_data[0])

        # split indexes
        idx = dw4_data[1]

        # standardize the data
        all_data = all_data.reshape(-1, NUM_PARTICLES, DIM)
        all_data = all_data - all_data.mean(axis=1, keepdims=True)
        all_data = all_data / torch.tensor([DW4_STD])

        # rotation augementation
        x = torch.rand(len(all_data)) * 2 * np.pi
        s = torch.sin(x)
        c = torch.cos(x)
        rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).permute(2, 0, 1)
        all_data = torch.einsum("bij,bki->bkj", rot, all_data)

        # return to vector shape
        all_data = all_data.reshape(-1, NUM_PARTICLES * DIM)

        # split the data
        self.data_train = all_data[idx[: TRAIN_VAL_TEST_SPLIT[0]]]
        self.data_val = all_data[idx[TRAIN_VAL_TEST_SPLIT[0] : TRAIN_VAL_TEST_SPLIT[1]]]
        self.data_test = all_data[idx[TRAIN_VAL_TEST_SPLIT[2] :]]

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = DW4DataModule()
