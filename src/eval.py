import os
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import rootutils
import torch
from bgflow.bg import sampling_efficiency
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.accelerators.cpu import CPUAccelerator
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.dw4_plots import TARGET, distance_histogram, energy_histogram

torch.set_float32_matmul_precision("high")  # TODO can we use medium instead
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting generation!")
    outputs = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    samples_proposal, log_p_proposal, samples_prior = zip(*outputs)  # TODO rename log_p (proposal)

    samples_proposal = torch.cat(samples_proposal, dim=0)
    samples_prior = torch.cat(samples_prior, dim=0)
    log_p_proposal = torch.cat(log_p_proposal, dim=0)

    outputs_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/npy_outputs"
    os.makedirs(outputs_dir)

    np.save(outputs_dir + "/samples_proposal", samples_proposal.cpu().numpy())
    np.save(outputs_dir + "/samples_prior", samples_prior.cpu().numpy())
    np.save(outputs_dir + "/log_p_proposal", log_p_proposal.cpu().numpy())

    metric_dict = trainer.callback_metrics

    # destandardize samples
    samples_proposal = samples_proposal.view(-1, 4, 2)
    samples_proposal *= torch.tensor([1.8230, 1.8103])
    samples_proposal = samples_proposal.view(-1, 8)

    # compute importance weights

    logits = -TARGET.energy(samples_proposal).flatten() - log_p_proposal.flatten()
    max_logits = torch.max(logits)
    importance_weights = torch.nn.functional.softmax(logits - max_logits)

    log.info(f"Sampling efficiency: {sampling_efficiency(logits).item()}")  # TODO properly log

    plots_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/plots"
    os.makedirs(plots_dir)

    energy_histogram(samples_proposal, importance_weights, save_path="latest_energy_histogram.png")
    distance_histogram(
        samples_proposal, importance_weights, save_path="latest_distance_histogram.png"
    )
    energy_histogram(
        samples_proposal, importance_weights, save_path=plots_dir + "/energy_histogram.png"
    )
    distance_histogram(
        samples_proposal, importance_weights, save_path=plots_dir + "/distance_histogram.png"
    )

    model = model.to(trainer.strategy.root_device)  # TODO won't handle multi-gpu

    samples_jarzynski, jarzynski_weights = model.jarzyinski_process(samples_proposal)

    log.info(
        f"Sampling efficiency: {sampling_efficiency(np.log(jarzynski_weights)).item()}"
    )  # TODO properly log

    plots_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/plots"

    energy_histogram(
        samples_proposal,
        importance_weights,
        samples_jarzynski,
        jarzynski_weights,
        save_path="latest_energy_histogram.png",
    )
    distance_histogram(
        samples_proposal,
        importance_weights,
        samples_jarzynski,
        jarzynski_weights,
        save_path="latest_distance_histogram.png",
    )
    energy_histogram(
        samples_proposal,
        importance_weights,
        samples_jarzynski,
        jarzynski_weights,
        save_path=plots_dir + "/energy_histogram.png",
    )
    distance_histogram(
        samples_proposal,
        importance_weights,
        samples_jarzynski,
        jarzynski_weights,
        save_path=plots_dir + "/distance_histogram.png",
    )

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
