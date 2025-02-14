if [ -z "$1" ]
  then
    echo "No argument supplied"
    exit 1
fi
if [ $1 = 2 ]; then
    python src/train.py \
    experiment=aldp logger=wandb \
    data=aldp \
    trainer=gpu trainer.max_epochs=1000 \
    tags=[al,cnf,eval,v9] \
    data.batch_size=512 \
    model.sampling_config.batch_size=256 \
    model.sampling_config.num_test_proposal_samples=10000 \
    model.net.hidden_nf=256 \
    data.num_workers=2 \
    model.div_estimator="ito" \
    train=False \
    ckpt_path='${oc.env:AL2_EQ1}/last.ckpt'
        #,${oc.env:AL2_EQ2}/last.ckpt,${oc.env:AL2_EQ3}/last.ckpt'
elif [ $1 = 3 ]; then
    python src/train.py -m \
    experiment=aldp logger=wandb \
    data=al3 \
    trainer=gpu trainer.max_epochs=10 \
    tags=[al,cnf,eval,v9] \
    data.batch_size=512 \
    +trainer.num_sanity_val_steps=0 \
    model.sampling_config.batch_size=32 \
    model.sampling_config.num_test_proposal_samples=10000 \
    model.net.hidden_nf=256 \
    data.num_workers=2 \
    train=True \
    #ckpt_path='${oc.env:AL3_EQ3}/last.ckpt'
    #ckpt_path='${oc.env:AL3_EQ1}/last.ckpt,${oc.env:AL3_EQ2}/last.ckpt,${oc.env:AL3_EQ3}/last.ckpt'
elif [ $1 = 4 ]; then
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
elif [ $1 = 5 ]; then
    python src/train.py -m \
    experiment=aldp logger=wandb \
    data=al5 \
    trainer=ddp trainer.max_epochs=1000 \
    tags=[al,cnf,eval,v9] \
    data.batch_size=512 \
    model.sampling_config.batch_size=32 \
    model.sampling_config.num_test_proposal_samples=1000 \
    model.net.hidden_nf=256 \
    data.num_workers=2 \
    train=False \
    +model.net.pdb_filename='${data.pdb_filename}' \
    ckpt_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/runs/al5_eqcnf_v3/checkpoints/last.ckpt"
elif [ $1 = 6 ]; then
    python src/train.py -m \
    experiment=aldp logger=wandb \
    data=al6 \
    trainer=ddp trainer.max_epochs=1000 \
    tags=[al,cnf,eval,v9] \
    data.batch_size=512 \
    model.sampling_config.batch_size=32 \
    model.sampling_config.num_test_proposal_samples=1000 \
    model.net.hidden_nf=256 \
    data.num_workers=2 \
    seed=1,2 \
    train=False \
    +model.net.pdb_filename='${data.pdb_filename}' \
    ckpt_path="/network/scratch/a/alexander.tong/fast-tbg//logs//train/runs/al6_eqcnf_v3/checkpoints/last.ckpt"
fi
