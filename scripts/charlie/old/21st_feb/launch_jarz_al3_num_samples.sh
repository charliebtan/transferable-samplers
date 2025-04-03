#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=jarz_tarflow_al3 \
trainer=gpu \
tags=[jarz,jarz_num_samples_v2] \
model.sampling_config.num_test_proposal_samples=100,200,500,1_000,2_000,5_000,10_000,20_000,50_000,100_000,200_000 \
model.jarzynski_sampler.num_timesteps=100 \
model.sampling_config.energy_cutoff=-120.0 \
model.jarzynski_sampler.langevin_eps=1e-7 \
model.jarzynski_sampler.ess_threshold=0.5 \
model.jarzynski_sampler.batch_size=1024 \
model.clip_logits=null,0.002 \
model.use_com_energy=1,0 \
seed=0,1 \
ckpt_path="/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-57-11/2/checkpoints/epoch_899_cropped.ckpt","/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-57-11/0/checkpoints/epoch_949_cropped.ckpt","/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-57-11/1/checkpoints/epoch_999_cropped.ckpt" \
