#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=aldp trainer=gpu train=False \
tags=[cnf,aldp,jarz] \
python src/train.py data=aldp model=normalizing_flow ckpt_path='${oc.env:AL2_TAR_NEW}' train=false trainer=gpu model.sampling_config.batch_size=10000 model.net.in_channels=1 model.net.num_blocks=6 model.jarzynski_sampler.enabled=True model.jarzynski_sampler.batch_size=256 model.sampling_config.num_jarzynski_samples=1024