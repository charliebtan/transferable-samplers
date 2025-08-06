#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=evaluation/ecnf_2aa \
logger=wandb \
model.eval_seq_name="AC","RV" \
seed="range(0, 20)" \
+model.sample_set=null \
ckpt_path=/network/scratch/t/tanc/ecnf_2aa_v1.ckpt \
model.sampling_config.num_test_proposal_samples=500