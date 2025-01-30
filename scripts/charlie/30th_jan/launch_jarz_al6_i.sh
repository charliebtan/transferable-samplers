#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
python src/train.py -m \
experiment=jarz_tarflow_al6 \
trainer=gpu \
tags=[jarz,jarz_big] \
model.jarzynski_sampler.num_timesteps=100 \
model.sampling_config.energy_cutoff=-20.0 \
model.jarzynski_sampler.langevin_eps=1e-7 \
model.jarzynski_sampler.ess_threshold=0.5 \
model.use_com_energy=1,0 \
seed=0,1,2