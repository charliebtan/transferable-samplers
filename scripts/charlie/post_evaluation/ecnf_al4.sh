#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=training/tarflow_al4 \
tags=[ecnf,al4,post_eval] \
+model.sampling_config.load_samples_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-03-29_07-26-20/2/test_samples.pt","/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-03-29_07-26-20/3/test_samples.pt","/network/scratch/a/alexander.tong/fast-tbg/logs/train/runs/2025-03-29_07-23-09/test_samples.pt" \
train=False 
