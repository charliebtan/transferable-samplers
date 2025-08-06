sequences=(
    AC
#     RV
)

# python src/train.py -m experiment=evaluation/tarflow_up_to_8aa logger=wandb tags=[tbg_leon_eval_v1] model.eval_seq_name=AC +model.sample_set=tbg_leon_samples +model.sampling_config.use_com_adjustment=False

# Pick the sequence based on SLURM_ARRAY_TASK_ID
seq=${sequences[$SLURM_ARRAY_TASK_ID]}

python src/train.py -m \
    experiment=evaluation/tarflow_up_to_8aa \
    logger=wandb \
    tags=[tbg_leon_eval_v1] \
    model.eval_seq_name="$seq" \
    +model.dont_fix_chirality=False \
    +model.energy_maxiter=100 \
    +model.sample_set=tbg_leon