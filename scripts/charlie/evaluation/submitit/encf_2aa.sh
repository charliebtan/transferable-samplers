#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=evaluation/ecnf_2aa \
tags=[2aa,ecnf_post_eval_v7] \
logger=wandb \
model.eval_seq_name="AC","AT","ET","GN","GP","HT","IM","KG","KQ","KS","LW","NF","NY","RL","RV","TD" \
+model.dont_fix_symmetry=True \