#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=jarz_tarflow_aldp \
trainer=gpu \
tags=[jarz,jarz_sweep_v2] \
model.jarzynski_sampler.num_timesteps=100,50,20,200,500 \
model.sampling_config.energy_cutoff=20.0,0.0 \
model.jarzynski_sampler.langevin_eps=2e-7,5e-7,1e-6,2e-6,5e-6,1e-5 \
model.jarzynski_sampler.ess_threshold=0.9,0.8,0.7,0.5 \
model.sampling_config.batch_size=20000 \
model.jarzynski_sampler.batch_size=1024 \
ckpt_path="/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/12/checkpoints/epoch_699_cropped.ckpt","/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/4/checkpoints/epoch_899_cropped.ckpt"

# ckpt_path=/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/12/checkpoints/epoch_699_cropped.ckpt # wobbly-microwave-2209
# ckpt_path=/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/4/checkpoints/epoch_899_cropped.ckpt # classic-smoke-2196