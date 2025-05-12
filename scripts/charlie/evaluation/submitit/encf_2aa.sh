#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=evaluation/ecnf_2aa \
tags=[2aa,ecnf_eval_v2] \
logger=wandb \
model.eval_seq_name="AC","AT","ET","GN","GP","HT","IM","KG","KQ","KS","LW","NF","NY","RL","RV","TD" \
seed="range(0, 10)" \
model.sampling_config.num_test_proposal_samples=1_000