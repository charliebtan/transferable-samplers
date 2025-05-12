#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=evaluation/tarflow_up_to_4aa \
tags=[up_to_4aa,eval,ablation_v2,pair_bias] \
logger=wandb \
model.net.use_attn_pair_bias=True \
model.eval_seq_name="AC","AT","ET","GN","GP","HT","IM","KG","KQ","KS","LW","NF","NY","RL","RV","TD","SAEL","RYDT","CSFQ","FALS","CSGS","LPEM","LYVI","AYTG","VCVS","AAEW","FKVP","NQFM","DTDL","CTSA","ANYT","VTST","AWKC","RGSP","AVEK","FIYG","VLSM","QADY","DQAL","TFFL","FIGE","KKQF","SLTC","ITQD","DFKS","QDED" \
model.sampling_config.num_test_proposal_samples=250_000 \
ckpt_path="/network/scratch/t/tanc/self-consume-bg/logs/train/runs/tarflow_up_to_4aa_pair_bias_small_v3/checkpoints/last.ckpt"
