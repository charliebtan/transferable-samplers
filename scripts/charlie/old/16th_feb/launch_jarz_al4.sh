#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=jarz_tarflow_al4 \
trainer=gpu \
tags=[jarz,jarz_final_iv10] \
model.sampling_config.num_test_proposal_samples=10_000,100_000 \
model.jarzynski_sampler.num_timesteps=100 \
model.sampling_config.energy_cutoff=50 \
model.jarzynski_sampler.langevin_eps=1e-7 \
model.jarzynski_sampler.ess_threshold=0.5 \
model.jarzynski_sampler.batch_size=1024 \
model.use_com_energy=1 \
ckpt_path="/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-56-57/0/checkpoints/epoch_749_cropped.ckpt","/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-56-57/1/checkpoints/epoch_999_cropped.ckpt","/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-56-57/2/checkpoints/epoch_999_cropped.ckpt" \
