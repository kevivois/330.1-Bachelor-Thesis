# 330.1-Bachelor-Thesis

This repository contains the code used for data preprocessing, feature extraction, visualization, and model training for the bachelor thesis project.

## Overview

This project implements a machine learning pipeline for predictive maintenance using sensor and torque data.
It includes feature extraction with TSFEL, visualization tools, and two deep learning models (LSTM, TFT).

## Project Structure

- `code/` — main source code
- `code/models/` — model definitions and training utilities
- `data`- containing multiple versions of pre-processed data (*.csv)
- `docs` - containing all admin reports and notes & meetings
- `results` - containing results of trained models and some graphs

## How to Run

```bash
cd code/
```

```bash
uv venv .venv
```

```bash
uv pip sync requirements.txt
```


## Visualization & Analysis

| Script | Purpose |
|---|---|
| `pca.py` | 3D PCA projections by tool and pass type |
| `t-sne.py` | 2D t-SNE |
| `spearman-barplot.py` | Top correlated features per tool |
| `spectral_centroid.py` | Sound spectral centroid evolution |
| `pass-trend.py` | Target variable trend with pass markers |


### How to configure & run a training

in `code/train-k-fold-lstm.py` modifiy this area for your needs

```python

datasets = [
        (csv_path,output_result_path)
    ]
    
    configs = [
        # model_type, with_exo, with_sound, only_sound, with_optuna, passes
    ("LSTM", True,  True,  False, False, ["Finishing"]),
    ("LSTM", True,  False, False, False, ["Finishing"]),
    ("LSTM", False, True,  False, False, ["Finishing","Pre-Finishing"]),
    ("LSTM", False, False, False, False, ["Finishing"]),
    ("LSTM", False, True,  True,  False, ["Finishing"])

    ]        

```
> replace `csv_path` by the pre-processed csv data and `output_result_path` by the output result folder
> in the `config` variable, you can configure a bunch of training that wil be executed with differents parameters , each one linked in the commentary


| parameter name | Purpose|
|---|---|
| `model type` | name/type of the model (will be the name in the output json) |
| `with_exo` | if true : will calcule some exogenous variables, columns name start with next_xxx |
| `with_sound` | if true: the sounds sensor column will be part of the features|
| `only_sound` | if true: the features will be only the sound columns|
| `with_optuna` | if true : each run will call the optuna hyperparameters optimization and then start a train with the best params |
| `passes`| an array of the type of passes we want in our training data


One the parameters configured , either start the according slurm file using `sbatch file.slurm` or directly using uv : `uv run train-xxxx.py`


### How to configure & run a data pre-processing

In the `main` function at the bottom of the `pre-processing.py` file , modifiy according your needs :

```python

basepath = "/home/kevin.voisin/datasets/kevivois/data_new/" # the base-folder of the parquet files
target =  "Broche/StatusTorqueData.ActualTorque_Median" # the target that will be shifted and in a new column named 'y'
output_filename = "tsfel_extracted_v5" # the output csv filename

```

then either use `uv run pre-processing.py` or using slurm : `sbatch pre-processing.slurm`

It outputs two CSV files :
- One with all features (complete)
- One after tsfresh feature selection (reduced)
They both includes aligned sensor features, torque features, metadata, and the targeted variable.


## Dependencies

See `code/requirements.txt` for the full list.


## Author

Kevin Voisin — HES-SO Valais
Bachelor Thesis 330.1