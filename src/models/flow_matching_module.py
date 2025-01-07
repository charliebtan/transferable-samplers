from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class FlowMatchLitModule(LightningModule):
    """

    TODO - Add a description.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `FlowMatchLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_loss = torch.nn.MSELoss()

        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(8), torch.eye(8)
        ) # TODO is this the right place for this?

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x:
        :param t: 
        :return: dx
        """
        # vt = bg.flow._dynamics._dynamics._dynamics_function(t, x)
        return self.net(x, t)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()
        pass

    def model_step(
        self, batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch:

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
        """

        x_1 = batch
        x_0 = self.prior.sample((x_1.shape[0],)).to(x_1.device)
        t = torch.rand(x_1.shape[0], 1, device=x_1.device) # should this be generated here or elsewhere?

        x_t = (1.0 - (1.0 - 1e-5) * t) * x_0 + t * x_1
        v_t_ref = x_1 - (1.0 - 1e-5) * x_0

        v_t_pred = self.forward(x_t, t)
        loss = self.criterion(v_t_pred, v_t_ref)

        # TODO the notebook had what looked like a diffusion target?
        # mu_t = x0 * (1 - t) + x1 * t
        # sigma_t = sigma
        # noise = prior.sample(batchsize)
        # x = mu_t + sigma_t * noise

        return loss, v_t_pred

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = FlowMatchLitModule(None, None, None, None)
