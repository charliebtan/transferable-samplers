
python src/train.py \
  experiment=aldp model/net=dit trainer=ddp  \
  data.batch_size=1024 +trainer.precision=bf16 \
  ++data.com_augmentation=True data=al4 model.sampling_config.batch_size=40 \
  model.sampling_config.num_proposal_samples=120 model.mean_free_prior=false \
  +trainer.num_sanity_val_steps=0 data.pin_memory=False \
  trainer.strategy=ddp_find_unused_parameters_true \
  callbacks.model_checkpoint.monitor=null \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_epochs=2 \
  callbacks.model_checkpoint.save_on_train_epoch_end=True \
  callbacks.model_checkpoint.verbose=True
