#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
python src/train.py -m \
experiment=jarz_tarflow_al3 \
trainer=gpu \
tags=[jarz,jarz_rs_v2,lambda] \
model.jarzynski_sampler.num_timesteps=100 \
model.sampling_config.energy_cutoff=-120.0 \
model.jarzynski_sampler.langevin_eps=1e-7 \
model.use_com_energy=1,0 \
model.jarzynski_sampler.ess_threshold=0.9,0.7,0.5,0.3,0.1,0.0 \
model.jarzynski_sampler.batch_size=3076 \
ckpt_path="/home/ubuntu/scratch/al3/0.ckpt","/home/ubuntu/scratch/al3/1.ckpt","/home/ubuntu/scratch/al3/2.ckpt" \
