#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=training/tarflow_al6 \
tags=[ecnf++,al6,post_eval] \
+model.sampling_config.load_samples_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/runs/al6_cfm_v2_v21_run2/test_samples.pt","/network/scratch/a/alexander.tong/fast-tbg/logs/train/runs/al6_cfm_v2_v21_run3/test_samples.pt" \
train=False 
