#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=training/tarflow_al3 \
tags=[al3,tarflow,refactor_test] \
logger=wandb \
trainer=gpu \
trainer.max_epochs=2000 \
data.batch_size=256 \
seed=0,1,2
