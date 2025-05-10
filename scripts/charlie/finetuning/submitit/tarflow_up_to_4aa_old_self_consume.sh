#!/bin/bash
python src/finetune.py -m launcher=mila_48gb \
experiment=finetuning/tarflow_up_to_4aa_old_self_consume \
tags=[up_to_4aa,self_consume,eval] \
logger=wandb \
model.eval_seq_name="RSCR","TAQE","RGGF","TGRC","CWVY" \
