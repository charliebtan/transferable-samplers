#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
python src/train.py -m \
experiment=jarz_tarflow_aldp \
trainer=gpu \
tags=[jarz,jarz_final_iv5] \
model.sampling_config.num_test_proposal_samples=1_000,10_000,100_000,1_000_000 \
model.jarzynski_sampler.num_timesteps=100 \
model.sampling_config.energy_cutoff=10.0 \
model.jarzynski_sampler.langevin_eps=1e-7 \
model.jarzynski_sampler.ess_threshold=0.5 \
model.jarzynski_sampler.batch_size=3076 \
model.use_com_energy=1,0 \
ckpt_path="/home/ubuntu/scratch/al2/0.ckpt","/home/ubuntu/scratch/al2/1.ckpt","/home/ubuntu/scratch/al2/2.ckpt" \

# ckpt_path=/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/12/checkpoints/epoch_699_cropped.ckpt # wobbly-microwave-2209
# ckpt_path=/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/4/checkpoints/epoch_899_cropped.ckpt # classic-smoke-2196
