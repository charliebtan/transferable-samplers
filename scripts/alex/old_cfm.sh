python src/train.py experiment=ecnf_al3 data=aldp \
  trainer=gpu trainer.check_val_every_n_epoch=100  \
  train=False \
  ckpt_path=/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-02-15_11-32-09/0/checkpoints/epoch_999.ckpt \
  model.sampling_config.num_test_proposal_samples=100 
#python src/train.py -m experiment=ecnf_al3 launcher=mila_48gb_long data=al3 \
#  trainer=gpu trainer.check_val_every_n_epoch=100 seed=42,43,44 tags="[old_cfm,v3]"
