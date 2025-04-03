#!/bin/bash
python src/train.py -m launcher=mila_48gb \
tags=[cnf,aldp,jarz,v2] \
data=aldp model=normalizing_flow ckpt_path='${oc.env:AL2_TAR}' train=false trainer=gpu model.sampling_config.batch_size=10000 model.net.in_channels=1 model.net.num_blocks=6 model.jarzynski_sampler.enabled=True model.jarzynski_sampler.batch_size=256 model.sampling_config.num_jarzynski_samples=1024 \
model.jarzynski_sampler._target_=src.models.components.jarzynski_sampler.JarzynskiSampler,src.models.components.fast_jarzynski_sampler.FastJarzynskiSampler \
model.jarzynski_sampler.langevin_eps=0.1,0.05,0.01,0.005 \
model.jarzynski_sampler.num_timesteps=1000,100,10
