#!/bin/bash
#SBATCH --job-name=jsd
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 1:00:00
#SBATCH --partition=long,main,unkillable
#SBATCH --array=0-76
#SBATCH --output=logs/jsd_%A_%a.out
#SBATCH --error=logs/jsd_%A_%a.err

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
    PGESTAES
    NKEKFFQH
    MYGRNCYM
    IDHRQLKW
    HWHSLICK
    NPCLCYML
    MRDPVLFA
    DDRDTEQT
    YFPHAGYT
    ISKCKNGE
    KRRGFFLE
    CLCCGQWN
    GNDLVTVI
    EKYYWMQT
    FWRVDHDM
    DGVAHALS
    PLFHVMYV
    SQQKVAFE
    IFGWVYTG
    CGSWHKQR
    WTYAFAHS
    MWNSTEMI
    PYIRNCVE
    ANKSMIEA
    MAPQTIAT
    SPHKMRLC
    VWIPVIDT
    NHQYGSDP
    PPWRECNN
)

# Pick the sequence based on SLURM_ARRAY_TASK_ID
seq=${sequences[$SLURM_ARRAY_TASK_ID]}

python src/train.py -m \
    experiment=evaluation/tarflow_up_to_8aa \
    logger=wandb \
    tags=[jsd,base10k_jsd_v1] \
    model.eval_seq_name="$seq" \
    +model.sample_set=base10k

python src/train.py -m \
    experiment=evaluation/tarflow_up_to_8aa \
    logger=wandb \
    tags=[jsd,full10k_jsd_v1] \
    model.eval_seq_name="$seq" \
    +model.sample_set=full10k