#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=evaluation/tarflow_up_to_4aa_hmc \
trainer=gpu \
tags=[sampling,eval_tarflow_up_to_4aa,smc,hmc,dev] \
data.train_lmdb_prefix="train_medium" \
data.val_lmdb_prefix=val \
data.test_lmdb_prefix=test \
model.smc_sampler.num_timesteps=100 \
model.smc_sampler.langevin_eps=1e-8 \
model.smc_sampler.ess_threshold=0.9 \
model.smc_sampler.batch_size=256 \
+model.smc_sampler.systematic_resampling=True \
model.smc_sampler.input_energy_cutoff=100 \
seed=0 \
ckpt_path="/network/scratch/m/majdi.hassan/self-consume-bg/self-consume-bg/logs/train/runs/tarflow_up_to_4aa_v2/checkpoints/last.ckpt" \
+model.eval_seq_name="RSCR","TAQE","RGGF","TGRC","CWVY"