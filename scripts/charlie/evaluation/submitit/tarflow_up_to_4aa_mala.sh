#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=evaluation/tarflow_up_to_4aa_mala \
trainer=gpu \
tags=[eval_tarflow_up_to_4aa,sampling_v1,smc,mala] \
data.train_lmdb_prefix="train_medium" \
train=False \
val=False \
test=True \
data.val_lmdb_prefix=val \
data.test_lmdb_prefix=test \
model.smc_sampler.batch_size=256 \
model.smc_sampler.input_energy_cutoff=100 \
model.eval_seq_name="AC","AT","ET","GN","GP","HT","IM","KG","KQ","KS","LW","NF","NY","RL","RV","TD","SAEL","RYDT","CSFQ","FALS","CSGS","LPEM","LYVI","AYTG","VCVS","AAEW","FKVP","NQFM","DTDL","CTSA","ANYT","VTST","AWKC","RGSP","AVEK","FIYG","VLSM","QADY","DQAL","TFFL","FIGE","KKQF","SLTC","ITQD","DFKS","QDED"  \
seed=0 \
ckpt_path="/network/scratch/m/majdi.hassan/self-consume-bg/self-consume-bg/logs/train/runs/tarflow_up_to_4aa_v2/checkpoints/last.ckpt" \
