module purge && \
module load python/3.11 && \
module load cuda/12.2 && \
module load openmm/8.2.0 && \
mkdir -p ~/envs && \
cd ~/envs/ && \
virtualenv --no-download tbg3 && \
source ~/envs/tbg3/bin/activate && \
cd ~/self-consume-bg/ && \
pip install -r requirements_tamia.txt