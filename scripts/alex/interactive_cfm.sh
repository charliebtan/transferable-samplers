
python src/train.py -m \
experiment=aldp logger=wandb launcher=mila_alex \
seed=42,43,44 \
data=aldp,al3,al4 \
trainer=gpu trainer.max_epochs=1000 \
model/net=egnn_dynamics_ad2_cat_v2 \
tags=[al,cnf,v9,egnn_v2] \
trainer.check_val_every_n_epoch=200 \
data.batch_size=512 \
model.sampling_config.batch_size=20 \
+trainer.num_sanity_val_steps=0 \
model.sampling_config.num_proposal_samples=1000 \
model.net.hidden_nf=256 \
#trainer.strategy=ddp_find_unused_parameters_true \
