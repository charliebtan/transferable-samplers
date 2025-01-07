# Fast Transferable Boltzmann Generators

## Install
```
conda create -n fast-tbg python=3.11
conda activate fast-tbg
pip install -r requirements.txt
```

## Train
```
python src/train.py trainer=cpu
```

## Sample Proposal
```
python src/eval.py ckpt_path=${CHECKPOINT_PATH} data.n_samples=4196 data.batch_size=256
```

## Legacy README


You also need things from the Transferrable Boltzmann Generator codebase.
https://osf.io/n8vz3/?view_only=1052300a21bd43c08f700016728aa96e

Would clone this for data and models.



## Datasets
Datasets and model checkpoints are available for training the Transferable Boltzmann Generators. To compare the performance of these generators on dipeptides against extended MD simulations, we used the 2AA-1-huge dataset. You can find it here: [timewarp datasets](https://huggingface.co/datasets/microsoft/timewarp).

## Code
This repository includes Python scripts and Jupyter notebooks that reproduce the main experiments detailed in our paper.

### Requirements
The necessary packages are listed in `requirements.txt`. You will also need to install:
- [bgflow](https://github.com/noegroup/bgflow)
- [bgmol](https://github.com/noegroup/bgmol/tree/main)

## Alanine Dipeptide

### Training Scripts
- **TBG + Backbone:** `AD2_classical_train_backbone.py`
- **TBG + Full:** `AD2_classical_train_tbg_full.py`

### Sampling Scripts
- **TBG + Backbone:** `AD2_classical_sample_backbone.py`
- **TBG + Full:** `AD2_classical_sample_tbg_full.py`

### Evaluation Notebook
Evaluate sampling results with: `AD2_evaluation.ipynb`.

## Other Dipeptides

### Training Scripts
- **TBG:** `2AA_train_tbg.py`
- **TBG + Backbone:** `2AA_train_tbg_backbone.py`
- **TBG + Full:** `2AA_train_tbg_full.py`

These training scripts do not require additional arguments.

### Sampling Scripts
- **TBG:** `2AA_sample_tbg.py`
- **TBG + Backbone:** `2AA_sample_tbg_backbone.py`
- **TBG + Full:** `2AA_sample_tbg_full.py`

These sampling scripts require the dipeptide abbreviation as an argument. For instance, to generate samples with the TBG + Full model for the dipeptide NY, use:
```bash
python 2AA_sample_tbg_full.py NY
