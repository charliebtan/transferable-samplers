#!/bin/bash
python src/train.py -m launcher=mila_48gb_short \
experiment=jarz_tarflow_al4 \
trainer=gpu \
tags=[jarz,jarz_rs] \
model.jarzynski_sampler.num_timesteps=100 \
model.sampling_config.energy_cutoff=50 \
model.jarzynski_sampler.langevin_eps=1e-7 \
model.jarzynski_sampler.ess_threshold=0.9,0.7,0.5,0.3,0.1,0.0 \
model.jarzynski_sampler.batch_size=1024 \
model.clip_logits=0.002 \
model.use_com_energy=1,0 \
ckpt_path="/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-56-57/0/checkpoints/epoch_749_cropped.ckpt","/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-56-57/1/checkpoints/epoch_999_cropped.ckpt","/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-56-57/2/checkpoints/epoch_999_cropped.ckpt" \
seed=0,1,2
