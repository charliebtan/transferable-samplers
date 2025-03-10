#!/bin/bash
#SBATCH -J train_al6                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=48G                     # server memory requested (per node)
#SBATCH -t 5-00:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=main
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100l:2                  # Type/number of GPUs needed
#SBATCH -c 4
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@90

RUN_NAME="al6_eqcnf_v3_seed_44"
#srun python -u src/train.py \
python -u src/train.py \
  experiment=aldp logger=wandb seed=44 data=al6 trainer=ddp \
model/net=egnn_dynamics_ad2_cat tags=[al,cnf,v12,egnn] \
  trainer.check_val_every_n_epoch=200 data.batch_size=512 \
  trainer.max_epochs=1000 \
  +model.net.pdb_filename='${data.pdb_filename}' \
  model.sampling_config.batch_size=40 +trainer.num_sanity_val_steps=1 \
  model.sampling_config.num_proposal_samples=20 model.net.hidden_nf=256 \
  model.sampling_config.num_test_proposal_samples=10000 \
  trainer.strategy=ddp_find_unused_parameters_true data.num_workers=6 \
hydra.run.dir='${paths.log_dir}/${task_name}/runs/'${RUN_NAME} \
ckpt_path='${paths.log_dir}/${task_name}/runs/'${RUN_NAME}/checkpoints/last.ckpt \
logger.wandb.id=${RUN_NAME}
