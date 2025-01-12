#!/bin/bash
python src/train.py -m launcher=mila_a100 \
model=invertible_shortcut logger=wandb \
trainer.check_val_every_n_epoch=10 trainer=gpu trainer.max_epochs=100 \
model.optimizer._target_=torch.optim.AdamW,torch.optim.Adam \
model.optimizer.weight_decay=0.01,0.0 model.scheduler.pct_start=0.1 \
tags=[dw4,tarflow,invert_shortcut,hparams1] model.d_base=0 \
model.base_flow_ckpt_path=/home/mila/a/alexander.tong/tbg/logs/train/runs/2025-01-11_20-44-27/checkpoints/last.ckpt 
