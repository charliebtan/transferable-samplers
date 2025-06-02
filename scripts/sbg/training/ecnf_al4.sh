#!/bin/bash
python src/train.py -m launcher=mila_rtx8000 \
experiment=training/ecnf_al4 \
tags=[sgb,al4,ecnf_sweep_v4] \
logger=wandb \
seed=0,1,2