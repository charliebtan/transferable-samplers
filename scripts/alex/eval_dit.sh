
#!/bin/bash
if [ -z "$1" ]
  then
    echo "No argument supplied"
    exit 1
fi
if [ $1 = 2 ]; then
  python src/train.py -m \
    experiment=aldp model/net=dit trainer=ddp  \
    data.batch_size=1024 +trainer.precision=bf16 \
    ++data.com_augmentation=True data=aldp model.sampling_config.batch_size=64 \
    model.sampling_config.num_proposal_samples=40 model.mean_free_prior=false \
    model.sampling_config.num_test_proposal_samples=10000 \
    +trainer.num_sanity_val_steps=0 data.pin_memory=False \
    model.atol=1e-5 \
    model.rtol=1e-5 \
    model.div_estimator="exact_no_functional" \
    trainer.check_val_every_n_epoch=200 \
    +model.strict_loading=False \
    tags=[al,dit,cnf,eval,v9] \
    train=False \
    ckpt_path='${oc.env:AL2_DIT2}/last.ckpt,${oc.env:AL2_DIT3}/last.ckpt'
elif [ $1 = 3 ]; then
  python src/train.py -m \
    experiment=aldp model/net=dit trainer=ddp  \
    data.batch_size=1024 +trainer.precision=bf16 \
    ++data.com_augmentation=True data=al3 model.sampling_config.batch_size=256 \
    model.sampling_config.num_proposal_samples=40 model.mean_free_prior=false \
    model.sampling_config.num_test_proposal_samples=10000 \
    +trainer.num_sanity_val_steps=0 data.pin_memory=False \
    model.atol=1e-5 \
    model.rtol=1e-5 \
    model.div_estimator="exact_no_functional" \
    trainer.check_val_every_n_epoch=200 \
    tags=[al,dit,cnf,eval,v9] \
    +model.dummy_ll=True \
    train=False \
    ckpt_path='${oc.env:AL3_DIT1}/last.ckpt,${oc.env:AL3_DIT2}/last.ckpt,${oc.env:AL3_DIT3}/last.ckpt'
    #ckpt_path='${oc.env:AL3_DIT3}/last.ckpt'
elif [ $1 = 4 ]; then
  python src/train.py -m \
    experiment=aldp model/net=dit trainer=ddp  \
    data.batch_size=1024 +trainer.precision=bf16 \
    ++data.com_augmentation=True data=al4 model.sampling_config.batch_size=256 \
    model.sampling_config.num_proposal_samples=40 model.mean_free_prior=false \
    model.sampling_config.num_test_proposal_samples=10000 \
    +trainer.num_sanity_val_steps=0 data.pin_memory=False \
    model.atol=1e-5 \
    model.rtol=1e-5 \
    model.div_estimator="exact_no_functional" \
    trainer.check_val_every_n_epoch=200 \
    tags=[al,dit,cnf,eval,v9] \
    +model.dummy_ll=True \
    train=False \
    ckpt_path='${oc.env:AL4_DIT1}/last.ckpt,${oc.env:AL4_DIT2}/last.ckpt,${oc.env:AL4_DIT3}/last.ckpt'
fi

