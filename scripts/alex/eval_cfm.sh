if [ False = True ]; then
    python src/train.py -m \
    experiment=aldp logger=wandb \
    data=aldp \
    trainer=gpu trainer.max_epochs=1000 \
    tags=[al,cnf,eval,v9] \
    data.batch_size=512 \
    model.sampling_config.batch_size=256 \
    model.sampling_config.num_test_proposal_samples=10000 \
    model.net.hidden_nf=256 \
    data.num_workers=2 \
    train=False \
    ckpt_path='${oc.env:AL2_EQ1}/last.ckpt,${oc.env:AL2_EQ2}/last.ckpt,${oc.env:AL2_EQ3}/last.ckpt'
fi
if [ False = True ]; then
    python src/train.py -m \
    experiment=aldp logger=wandb \
    data=al3 \
    trainer=gpu trainer.max_epochs=1000 \
    tags=[al,cnf,eval,v9] \
    data.batch_size=512 \
    model.sampling_config.batch_size=32 \
    model.sampling_config.num_test_proposal_samples=10000 \
    model.net.hidden_nf=256 \
    data.num_workers=2 \
    train=False \
    ckpt_path='${oc.env:AL3_EQ1}/last.ckpt,${oc.env:AL3_EQ2}/last.ckpt,${oc.env:AL3_EQ3}/last.ckpt'
fi
if [ True = True ]; then
    python src/train.py -m \
    experiment=aldp logger=wandb \
    data=al4 \
    trainer=ddp trainer.max_epochs=1000 \
    tags=[al,cnf,eval,v9] \
    data.batch_size=512 \
    model.sampling_config.batch_size=64 \
    model.sampling_config.num_test_proposal_samples=10000 \
    model.net.hidden_nf=256 \
    data.num_workers=2 \
    train=False \
    ckpt_path='${oc.env:AL4_EQ1}/last.ckpt,${oc.env:AL4_EQ2}/last.ckpt'
fi
