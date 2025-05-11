#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=evaluation/tarflow_up_to_4aa_old \
tags=[up_to_4aa,eval,sampling_v1,standard,is_only] \
logger=wandb \
model.eval_seq_name="AC","AT","ET","GN","GP","HT","IM","KG","KQ","KS","LW","NF","NY","RL","RV","TD","SAEL","RYDT","CSFQ","FALS","CSGS","LPEM","LYVI","AYTG","VCVS","AAEW","FKVP","NQFM","DTDL","CTSA","ANYT","VTST","AWKC","RGSP","AVEK","FIYG","VLSM","QADY","DQAL","TFFL","FIGE","KKQF","SLTC","ITQD","DFKS","QDED"  \
model.sampling_config.num_test_proposal_samples=1_000_000 \
ckpt_path="/network/scratch/m/majdi.hassan/self-consume-bg/self-consume-bg/logs/train/runs/tarflow_up_to_4aa_v2/checkpoints/last.ckpt" \

