#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=evaluation/tarflow_al3_fk \
tags=[sbg,tarflow,al3,smc_samples_v1] \
model.sampling_config.num_test_proposal_samples=10_000,100_000 \
model.sampling_config.use_com_adjustment=1 \
model.sampling_config.clip_reweighting_logits=0.002 \
model.smc_sampler.input_energy_cutoff=-120.0 \
model.smc_sampler.num_timesteps=100 \
model.smc_sampler.langevin_eps=1e-8,1e-7 \
model.smc_sampler.ess_threshold=0.5,-1.0 \
model.smc_sampler.batch_size=1024 \
ckpt_path="/home/mila/t/tanc/scratch/self-consume-bg/logs/train/multiruns/2025-05-30_04-42-59/0/checkpoints/epoch_599_resampled_energy_w2.ckpt","/home/mila/t/tanc/scratch/self-consume-bg/logs/train/multiruns/2025-05-30_04-42-59/0/checkpoints/epoch_799_resampled_energy_w2.ckpt","/home/mila/t/tanc/scratch/self-consume-bg/logs/train/multiruns/2025-05-30_04-42-59/0/checkpoints/epoch_999_resampled_energy_w2.ckpt"