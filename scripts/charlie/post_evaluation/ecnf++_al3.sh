#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=training/tarflow_al3 \
tags=[ecnf++,al3,post_eval] \
+model.sampling_config.load_samples_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-03-28_22-50-37/2/test_samples.pt","/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-03-28_22-50-37/1/test_samples.pt","/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-03-28_22-50-37/0/test_samples.pt" \
train=False 
