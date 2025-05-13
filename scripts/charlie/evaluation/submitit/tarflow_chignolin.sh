#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=evaluation/tarflow_scale \
trainer=gpu \
tags=[eval_tarflow_up_to_8aa,chignolin] \
model.eval_seq_name="GYDPETGTWG" \
seed=0 \
ckpt_path="/network/scratch/m/majdi.hassan/self-consume-bg/self-consume-bg/logs/train/runs/tarflow_up_to_8aa_v3/checkpoints/epoch_299_time.ckpt" \
