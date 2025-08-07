#!/bin/bash
#SBATCH --job-name=tbg_eval
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 1:00:00
#SBATCH --partition=long,main,unkillable
#SBATCH --array=0-16
#SBATCH --output=logs/tbg_%A_%a.out
#SBATCH --error=logs/tbg_%A_%a.err

sequences=(
    AC
    AT
    ET
    GN
    GP
    HT
    IM
    KG
    KQ
    KS
    LW
    NF
    NY
    RL
    RV
    TD
)

# Pick the sequence based on SLURM_ARRAY_TASK_ID
seq=${sequences[$SLURM_ARRAY_TASK_ID]}

python src/train.py -m \
    experiment=evaluation/tarflow_up_to_8aa \
    logger=wandb \
    model.eval_seq_name="$seq" \
    tags=[jsd,tbg_jsd_v1] \
    +model.sample_set=tbg_leon_samples \
    model.sampling_config.use_com_adjustment=False