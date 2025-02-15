python src/train.py -m experiment=ecnf_al3 launcher=mila_48gb_long data=aldp,al3,al4 \
  trainer=gpu trainer.check_val_every_n_epoch=100 seed=42,43,44 tags="[old_cfm,v2]"
