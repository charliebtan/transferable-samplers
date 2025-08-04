#!/bin/bash
#SBATCH --job-name=md_eval
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --partition=long
#SBATCH --array=0-31
#SBATCH --output=logs/md_%A_%a.out
#SBATCH --error=logs/md_%A_%a.err

sequences=(
    SAEL
    RYDT
    CSFQ
    FALS
    CSGS
    LPEM
    LYVI
    AYTG
    VCVS
    AAEW
    FKVP
    NQFM
    DTDL
    CTSA
    ANYT
    VTST
    AWKC
    RGSP
    AVEK
    FIYG
    VLSM
    QADY
    DQAL
    TFFL
    FIGE
    KKQF
    SLTC
    ITQD
    DFKS
    QDED
)

maxiters=(
    10000
)

# Pick the sequence based on SLURM_ARRAY_TASK_ID
seq=${sequences[$SLURM_ARRAY_TASK_ID]}

for maxiter in "${maxiters[@]}"; do
    python src/train.py -m \
        experiment=evaluation/tarflow_up_to_8aa \
        logger=wandb \
        tags=[md_eval_v3] \
        model.eval_seq_name="$seq" \
        +model.dont_fix_symmetry=True \
        +model.dont_fix_chirality=True \
        +model.energy_maxiter="$maxiter" \
        +model.sample_set=md
done