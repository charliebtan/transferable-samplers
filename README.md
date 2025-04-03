# Fast Transferable Boltzmann Generators

## Dev Setup

```
pip install ruff
pre-commit install
```

## Install
```
conda create -n fast-tbg python=3.11
conda activate fast-tbg
pip install -r requirements.txt
```

## Train
```
python src/train.py experiment=training/tarflow_aldp trainer=gpu
```

## Sampling
```
python src/eval.py ckpt_path=${CHECKPOINT_PATH} experiment=evaluation/tarflow_aldp
```
