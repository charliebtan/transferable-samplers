#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=aldp trainer=gpu train=False \
tags=[cnf,aldp,jarz,v2] \
ckpt_path=/home/mila/b/bosejoey/last-ema-alex.ckpt \
data.batch_size=1024 model.jarzynski_sampler.batch_size=64 \
model.sampling_config.batch_size=64 \
model.jarzynski_sampler.enabled=True \
model.sampling_config.num_jarzynski_samples=1024 \
model.jarzynski_sampler.langevin_eps=0.1,0.05,0.01,0.005 \
model.jarzynski_sampler.num_timesteps=1000,100,10
