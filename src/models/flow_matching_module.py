from typing import Any, Dict, Tuple

from bgflow import BoltzmannGenerator
from bgmol import MultiDoubleWellPotential
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric


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
        self.criterion = torch.nn.MSELoss(reduce="mean")

        # metric objects for calculating and averaging accuracy across batches
        self.train_loss = MeanMetric()

        ## TODO TODO I'm not sure this is the right place to have this

        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(8), torch.eye(8)
        ) # TODO is this the right place for this?

        # first define system dimensionality and a target energy/distribution
        dim = 8
        n_particles = 4
        n_dimensions = dim // n_particles

        # DW parameters
        a=0.9
        b=-4
        c=0
        offset=4

        self.target = MultiDoubleWellPotential(dim, n_particles, a, b, c, offset, two_event_dims=False)

        # having a flow and a prior, we can now define a Boltzmann Generator

        self.bg = BoltzmannGenerator(self.prior, self.net, self.target)

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
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch:

        :return: - A tensor of losses.
        """

        x1 = batch
        x0 = self.prior.sample((x1.shape[0],)).to(x1.device)
        t = torch.rand(x1.shape[0], 1, device=x1.device) # should this be generated here or elsewhere?

        xt = (1.0 - (1.0 - 1e-5) * t) * x0 + t * x1
        vt_ref = x1 - (1.0 - 1e-5) * x0

        vt_pred = self.forward(t, xt)
        loss = self.criterion(vt_pred, vt_ref)

        # TODO the notebook had what looked like a diffusion target?
        # mu_t = x0 * (1 - t) + x1 * t
        # sigma_t = sigma
        # noise = prior.sample(batchsize)
        # x = mu_t + sigma_t * noise

        return loss

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

    @torch.no_grad()
    def generate_samples(self, n_samples: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from the model.

        :param n_samples: The number of samples to generate.
        :return: A tuple containing the generated samples and their log weights.
        """
        n_batches = n_samples // batch_size

        sample_batches = []
        log_weights_batches = []

        for _ in range(n_batches):    
            samples, latent, dlogp = self.bg.sample(n_samples, with_latent=True, with_dlogp=True)
            log_weights = self.bg.log_weights_given_latent(samples, latent, dlogp, normalize=False)

            sample_batches.append(samples)
            log_weights_batches.append(log_weights)
        
        samples = torch.cat(sample_batches, dim=0)
        log_weights = torch.cat(log_weights_batches, dim=0)

        return samples, log_weights


if __name__ == "__main__":
    _ = FlowMatchLitModule(None, None, None, None)
