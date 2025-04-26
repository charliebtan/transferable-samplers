#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=evaluation/tarflow_2aa \
trainer=gpu \
tags=[sampling] \
model.smc_sampler.num_timesteps=1000 \
model.smc_sampler.langevin_eps=1e-8 \
model.smc_sampler.ess_threshold=0.9 \
model.smc_sampler.batch_size=1024 \
model.smc_sampler.input_energy_cutoff=null \
seed=0 \
ckpt_path="/home/mila/t/tanc/scratch/self-consume-bg/logs/train/runs/2aa_tarflow_v4/checkpoints/last.ckpt"

# you may need to set input_energy_cutoff to avoid NaN