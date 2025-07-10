#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=training/tarflow_al2 \
tags=[ecnf++,al2,post_eval] \
+model.sampling_config.load_samples_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-02-14_11-05-10/2/test_samples.pt","/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-02-14_11-05-10/1/test_samples.pt","/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-02-14_11-05-10/0/test_samples.pt" \
train=False 
