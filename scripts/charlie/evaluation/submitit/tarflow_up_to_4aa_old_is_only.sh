#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=evaluation/tarflow_up_to_4aa_old \
tags=[up_to_4aa,eval,is_only] \
logger=wandb \
model.eval_seq_name="RSCR","TAQE","RGGF","TGRC","CWVY" \
val=True \
test=False \
model.sampling_config.num_test_proposal_samples=10_000,100_000,1_000_000 \
model.sampling_config.clip_reweighting_logits=null,0.002
