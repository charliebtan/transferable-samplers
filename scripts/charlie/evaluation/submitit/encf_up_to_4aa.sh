#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=evaluation/ecnf_up_to_4aa \
tags=[up_to_4aa,ecnf_eval_v1] \
logger=wandb \
model.eval_seq_name="AC","AT","ET","GN","GP","HT","IM","KG","KQ","KS","LW","NF","NY","RL","RV","TD","SAEL","RYDT","CSFQ","FALS","CSGS","LPEM","LYVI","AYTG","VCVS","AAEW","FKVP","NQFM","DTDL","CTSA","ANYT","VTST","AWKC","RGSP","AVEK","FIYG","VLSM","QADY","DQAL","TFFL","FIGE","KKQF","SLTC","ITQD","DFKS","QDED" \
seed="range(0, 10)" \
model.sampling_config.num_test_proposal_samples=1_000