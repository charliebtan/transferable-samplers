#!/bin/bash
python src/train.py -m launcher=mila_a100l \
experiment=jarz_tarflow_chignolin \
trainer=gpu \
tags=[jarz,jarz_big_dev] \
model.jarzynski_sampler.num_timesteps=100 \
model.sampling_config.energy_cutoff=-100.0 \
model.sampling_config.num_test_proposal_samples=100_000 \
model.sampling_config.num_jarzynski_samples=100_000 \
model.jarzynski_sampler.langevin_eps=2e-10 \
model.jarzynski_sampler.ess_threshold=0.9,0.7,0.5 \
+model.jarzynski_sampler.do_energy_plots=True \
model.use_com_energy=1
