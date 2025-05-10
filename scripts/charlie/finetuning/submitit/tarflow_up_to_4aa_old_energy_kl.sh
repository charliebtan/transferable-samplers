#!/bin/bash
python src/finetune.py -m launcher=mila_48gb \
experiment=finetuning/tarflow_up_to_4aa_old_energy_kl \
tags=[up_to_4aa,energy_kl,eval] \
logger=wandb \
model.eval_seq_name="RSCR","TAQE","RGGF","TGRC","CWVY" \
model.energy_kl_weight=1e-3,1e-4,1e-5 
