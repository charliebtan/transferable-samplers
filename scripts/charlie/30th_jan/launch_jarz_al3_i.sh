#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
python src/train.py -m \
experiment=jarz_tarflow_al3 \
trainer=gpu \
tags=[jarz,jarz_final_iv5] \
model.sampling_config.num_test_proposal_samples=1_000,10_000,100_000,1_000_000 \
model.jarzynski_sampler.num_timesteps=100 \
model.sampling_config.energy_cutoff=-120.0 \
model.jarzynski_sampler.langevin_eps=1e-7 \
model.jarzynski_sampler.ess_threshold=0.5 \
model.jarzynski_sampler.batch_size=3076 \
model.use_com_energy=1,0 \
ckpt_path="/home/ubuntu/scratch/al3/0.ckpt","/home/ubuntu/scratch/al3/1.ckpt","/home/ubuntu/scratch/al3/2.ckpt" \
