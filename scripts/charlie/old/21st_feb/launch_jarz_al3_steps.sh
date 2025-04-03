#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=jarz_tarflow_al3 \
trainer=gpu \
tags=[jarz,jarz_steps_v3] \
model.sampling_config.num_test_proposal_samples=100_000 \
model.jarzynski_sampler.num_timesteps=10,20,50,100,200,500 \
model.sampling_config.energy_cutoff=-120.0 \
model.jarzynski_sampler.langevin_eps=1e-7 \
model.jarzynski_sampler.ess_threshold=0.5 \
model.jarzynski_sampler.batch_size=1024 \
model.use_com_energy=1,0 \
model.clip_logits=null \
ckpt_path="/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-57-11/2/checkpoints/epoch_899_cropped.ckpt","/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-57-11/0/checkpoints/epoch_949_cropped.ckpt","/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-57-11/1/checkpoints/epoch_999_cropped.ckpt" \
