python src/train.py -m launcher=mila_48gb \
experiment=aldp logger=wandb \
data=aldp \
trainer=gpu trainer.max_epochs=1000 \
tags=[al,cnf,v6] \
trainer.check_val_every_n_epoch=100 \
data.batch_size=512 \
model.sampling_config.batch_size=100 \
model.sampling_config.num_proposal_samples=1000 \
model.net.hidden_nf=256 \
seed=43 \
data.num_workers=2
#hydra.launcher.gres=gpu:a100l:1 \
#data=al3,al4 \
