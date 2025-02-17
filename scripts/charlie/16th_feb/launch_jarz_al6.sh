#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=jarz_tarflow_al6 \
trainer=gpu \
tags=[jarz,jarz_big_v3] \
model.sampling_config.num_test_proposal_samples=10_000,100_000 \
model.jarzynski_sampler.num_timesteps=100 \
model.sampling_config.energy_cutoff=-20.0 \
model.jarzynski_sampler.langevin_eps=1e-7 \
model.jarzynski_sampler.ess_threshold=0.5 \
model.jarzynski_sampler.batch_size=1024 \
model.use_com_energy=1 \
ckpt_path="/network/scratch/b/bosejoey/fast-tbg/logs/train/runs/tarflow_al6_v2/checkpoints/epoch_999_time.ckpt"