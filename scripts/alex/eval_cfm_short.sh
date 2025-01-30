#    python src/train.py \
#    experiment=aldp logger=wandb \
#    data=aldp \
#    trainer=gpu trainer.max_epochs=1000 \
#    tags=[al,cnf,eval_time,v9] \
#    data.batch_size=512 \
#    model.sampling_config.batch_size=64 \
#    model.sampling_config.num_test_proposal_samples=640 \
#    model.net.hidden_nf=256 \
#    data.num_workers=2 \
#    model.div_estimator="ito" \
#    train=False \
#    ckpt_path='${oc.env:AL2_EQ1}/last.ckpt'
#    python src/train.py -m \
#    experiment=aldp logger=wandb \
#    data=al3 \
#    trainer=gpu trainer.max_epochs=1000 \
#    tags=[al,cnf,eval_time,v9] \
#    data.batch_size=512 \
#    model.sampling_config.batch_size=64 \
#    model.sampling_config.num_test_proposal_samples=640 \
#    model.net.hidden_nf=256 \
#    data.num_workers=2 \
#    train=False \
#    ckpt_path='${oc.env:AL3_EQ3}/last.ckpt'
    python src/train.py -m \
    experiment=aldp logger=wandb \
    data=al4 \
    trainer=ddp trainer.max_epochs=1000 \
    tags=[al,cnf,eval_time,v9] \
    data.batch_size=512 \
    model.sampling_config.batch_size=32 \
    model.sampling_config.num_test_proposal_samples=320 \
    model.net.hidden_nf=256 \
    data.num_workers=2 \
    train=False \
    ckpt_path='${oc.env:AL4_EQ1}/last.ckpt'
