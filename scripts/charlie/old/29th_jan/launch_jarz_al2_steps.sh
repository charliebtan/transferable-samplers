#!/bin/bash
python src/train.py -m \
experiment=jarz_tarflow_aldp \
trainer=gpu \
tags=[jarz,jarz_steps] \
model.jarzynski_sampler.num_timesteps=10,20,50,100,200,500 \
model.sampling_config.energy_cutoff=10.0 \
model.jarzynski_sampler.langevin_eps=1e-7 \
model.jarzynski_sampler.ess_threshold=0.5 \
model.jarzynski_sampler.batch_size=3076 \
model.clip_logits=0.002 \
model.use_com_energy=0,1 \
ckpt_path="/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/12/checkpoints/epoch_699_cropped.ckpt","/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/4/checkpoints/epoch_899_cropped.ckpt","/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-24_07-35-01/0/checkpoints/epoch_699_cropped.ckpt" \
seed=0,1,2

# ckpt_path=/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/12/checkpoints/epoch_699_cropped.ckpt # wobbly-microwave-2209
# ckpt_path=/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/4/checkpoints/epoch_899_cropped.ckpt # classic-smoke-2196
