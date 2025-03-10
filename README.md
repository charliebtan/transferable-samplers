# Fast Transferable Boltzmann Generators

## Install
```
conda create -n fast-tbg python=3.11
conda activate fast-tbg
pip install -r requirements.txt
```

## Train
```
python src/train.py trainer=gpu
```

## Sampling
```
python src/eval.py ckpt_path=${CHECKPOINT_PATH} experiment=jarz_tarflow_al2
```

## Data Files
MD data files can be found below

https://osf.io/srqg7/files/osfstorage?view_only=af935a79a5e645b7aab5d37bc5eb3faa

https://osf.io/wm47v/files/osfstorage?view_only=af935a79a5e645b7aab5d37bc5eb3faa
