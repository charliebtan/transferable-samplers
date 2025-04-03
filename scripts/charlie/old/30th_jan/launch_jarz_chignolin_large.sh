#!/bin/bash
python src/train.py -m \
experiment=jarz_tarflow_chignolin \
trainer=ddp \
trainer.strategy=ddp_find_unused_parameters_true \
tags=[jarz,jarz_big] \
model.jarzynski_sampler.num_timesteps=100 \
model.sampling_config.energy_cutoff=-100.0 \
model.sampling_congig.num_test_proposal_samples=100_000 \
model.sampling_config.num_jarzynski_samples=100_000 \
model.sampling_config.batch_size=8192 \
model.jarzynski_sampler.batch_size=1024 \
model.jarzynski_sampler.langevin_eps=1e-9 \
model.jarzynski_sampler.ess_threshold=0.5 \
model.use_com_energy=1