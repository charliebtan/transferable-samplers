#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=training/tarflow_al4 \
tags=[ecnf++,al4,post_eval] \
+model.sampling_config.load_samples_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/runs/al4_cfm_v2_v22/test_samples.pt","/network/scratch/a/alexander.tong/fast-tbg/logs/train/runs/al4_cfm_v2_v21/test_samples.pt","/network/scratch/a/alexander.tong/fast-tbg/logs/train/runs/al4_cfm_v2_v20/test_samples.pt" \
train=False 
