#!/bin/bash
python src/train.py -m launcher=mila_rtx8000 \
experiment=training/ecnf_al3 \
tags=[sgb,al3,ecnf_sweep_v4] \
logger=wandb \
seed=0,1,2