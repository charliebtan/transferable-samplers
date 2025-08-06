#!/bin/bash
#SBATCH --job-name=unisim_eval
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --partition=long
#SBATCH --array=0-76
#SBATCH --output=logs/unisim_%A_%a.out
#SBATCH --error=logs/unisim_%A_%a.err

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
    tags=[my_tbg_eval_v1] \
    model.eval_seq_name="$seq" \
    +model.sample_set=ecnf_2aa \