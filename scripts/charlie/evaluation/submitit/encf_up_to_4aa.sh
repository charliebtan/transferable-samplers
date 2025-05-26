#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=evaluation/ecnf_up_to_4aa \
tags=[up_to_4aa,ecnf_post_eval_v6] \
logger=wandb \
model.eval_seq_name="SAEL","RYDT","CSFQ","FALS","CSGS","LPEM","LYVI","AYTG","VCVS","AAEW","FKVP","NQFM","DTDL","CTSA","ANYT","VTST","AWKC","RGSP","AVEK","FIYG","VLSM","QADY","DQAL","TFFL","FIGE","KKQF","SLTC","ITQD","DFKS","QDED","AC","AT","ET","GN","GP","HT","IM","KG","KQ","KS","LW","NF","NY","RL","RV","TD" \
+model.dont_fix_symmetry=True 
