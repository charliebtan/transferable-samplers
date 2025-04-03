#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
python src/train.py -m \
experiment=jarz_tarflow_al4 \
trainer=gpu \
tags=[jarz,jarz_final_iv5] \
model.sampling_config.num_test_proposal_samples=1_000,10_000,100_000,1_000_000 \
model.jarzynski_sampler.num_timesteps=100 \
model.sampling_config.energy_cutoff=50 \
model.jarzynski_sampler.langevin_eps=1e-7 \
model.jarzynski_sampler.ess_threshold=0.5 \
model.use_com_energy=1,0 \
ckpt_path="/home/ubuntu/scratch/al4/0.ckpt","/home/ubuntu/scratch/al4/1.ckpt","/home/ubuntu/scratch/al4/2.ckpt" \