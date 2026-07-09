# This script trains and evaluates a Temporal Fusion Transformer model using k-fold cross-validation.


import polars as pl
from models.TFTModel import TFTModel
from sklearn.model_selection import KFold
from pathlib import Path
import shutil
import json
import numpy as np
import random


'''
Function used to train the TFT Model from models.TFTModel using a k-fold tools split method
'''

def train(model_type="TFT", data_path="", with_exogenous_variables=True, with_sound_columns=True, only_sound_columns=False, with_optuna=False, n_pre_process_window=10, params={}, target="",passes = ["Finishing", "Pre-Finishing"],base_path=""):
    
    sound  = "sound"  if with_sound_columns       else "no_sound"
    exo    = "exo"    if with_exogenous_variables  else "no_exo"
    optuna = "optuna" if with_optuna               else "fixed"

    experiment_name = f"{model_type}_{optuna}_{sound}_{exo}"

    meta_cols = [    # meta columns
        "sensor_file", "timestamp", "time", "ToolIdx", "plate_id",
        "DB_PASSES/NUMERO_OF", "PassID", "start_pos", "end_pos",
        "DB_PASSES/NUMERO_PASSE", "timestamp_right", "y", "pass_type",
        "index", "DB_PASSES/COMPTEUR_RESET_PASSES"
    ]
    sound_prefixes = ["Sound", "AccZ", "AccY", "AccX"]   # Sound columns
    exogenous_variables = [ # Exogenous variables
        "DB_PASSES/SELECTION_ALLIAGE", "DB_PASSES/EPAISSEUR_BRUTE",
        "Axe_X_master/ActualVelocity_Mean",
        "Broche/ActualSpeed_Mean", "pass_type"
    ]
    dummies_cols = [ # Categorical columns
        "pass_type", "DB_PASSES/SELECTION_ALLIAGE",
        "next_pass_type", "next_DB_PASSES/SELECTION_ALLIAGE"
    ]

    data = (
        pl.scan_csv(data_path)
        .filter(pl.col("pass_type").is_in(passes))
        .collect()
    )

    if with_exogenous_variables:
        for c in exogenous_variables:
            data = data.with_columns(
                pl.col(c).shift(-1).over("ToolIdx").alias(f"next_{c}") # .over here to remove data that predict the next tool , remove tool-transition rows therefore data-leakage
            )
        data = data.drop_nulls(subset=[f"next_{c}" for c in exogenous_variables])

    existing_dummies = [c for c in dummies_cols if c in data.columns]
    data = data.to_dummies(columns=existing_dummies)

    feature_cols = [c for c in data.columns if c not in meta_cols]

    if only_sound_columns: # if true : the features will be only sound columns
        feature_cols = [
            c for c in feature_cols
            if any(c.startswith(p) for p in sound_prefixes)
        ]
    elif not with_sound_columns: # if true : sound columns will be removed of the features
        feature_cols = [
            c for c in feature_cols
            if not any(c.startswith(p) for p in sound_prefixes)
        ]

    cols_to_use = feature_cols + [c for c in meta_cols if c in data.columns]
    data = data.select(cols_to_use)

    kf = KFold(n_splits=4, shuffle=False)  #K-fold using 4 splits
    folder = Path(base_path + f"/k-fold-{experiment_name}-{TFTModel.get_formated_datetime()}")
    folder.mkdir(exist_ok=True, parents=True)

    tool_ids = data["ToolIdx"].unique().sort().to_list()
    cv_scores = {
        "rows": [],
        "summary": {},
        "header": {
            "with_sound_columns": with_sound_columns,
            "with_exogenous_variables": with_exogenous_variables,
            "only_sound_columns": only_sound_columns
        }
    }
    random.seed(42) # seed to ensure reproductibility
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X=tool_ids)):
        test_tools  = [tool_ids[i] for i in test_idx]
        train_tools = [tool_ids[i] for i in train_idx]
        
        # Randomly choose 2 tools for the validation tools set
        val_tools   = random.sample(train_tools, 2)
        train_tools = [t for t in train_tools if t not in val_tools]

        print(train_tools, val_tools, test_tools)

        model = TFTModel(
            data.lazy(),
            train_tools=train_tools,
            test_tools=test_tools,
            val_tools=val_tools,
            filepath=data_path,
            meta_cols=meta_cols,
            base_image_path=base_path,
            **params
        )

        if with_optuna:
            best_params = model.optimize_parameters(n_trials=10, n_epochs_optuna=25)
            print(f"best params: {best_params}")

        path = model.train()

        cv_scores["rows"].append({
            "fold":        fold_idx + 1,
            "r2":          model.res_r2,
            "rmse":        model.res_rmse,
            "mae":         model.res_mae,
            "train_tools": train_tools,
            "val_tools":   val_tools,
            "test_tools":  test_tools,
            "path":        str(path)
        })

    r2_list   = [row["r2"]   for row in cv_scores["rows"]]
    rmse_list = [row["rmse"] for row in cv_scores["rows"]]
    mae_list  = [row["mae"]  for row in cv_scores["rows"]]

    cv_scores["summary"] = {
        "mean_r2":   float(np.mean(r2_list)),
        "mean_rmse": float(np.mean(rmse_list)),
        "mean_mae":  float(np.mean(mae_list))
    }

    for row in cv_scores["rows"]:
        init_folder = Path(row["path"])
        if init_folder.exists():
            shutil.move(init_folder, folder / init_folder.name)

    result_file = folder / "data.json"
    result_file.write_text(json.dumps(cv_scores, indent=4))


if __name__ == "__main__":
    
    '''
    Training of the TFT model in differents configuration and parameters
    
    '''

    params = {
        "input_chunk_length":  20,
        "hidden_size":         32,
        "lstm_layers":         1,
        "num_attention_heads": 2,
        "batch_size":          32,
        "dropout":             0.1,
        "n_epochs":            50,
        "learning_rate":       0.001
    }
    
    datasets = [
        ("./tsfel_extracted_v5.csv", 10,"./runs/v6/10"),  # Using data that has beed reduced to 10 rows per pass file
        ("./tsfel_extracted_v8_y_mean_torque.csv", 1,"./runs/v6/1"),  # Using data that has been reduced to 1 row per pass file
    ]

    configs = [
        # model_type, with_exo, with_sound, only_sound, with_optuna, passes type3
        ("TFT", True,  True,  False, False, ["Finishing"]),
        ("TFT", True,  False, False, False, ["Finishing"]),
        ("TFT", True,  True,  False, False, ["Pre-Finishing"]),
        ("TFT", True,  False, False, False, ["Pre-Finishing"]),
        ("TFT", True,  True,  False, False, ["Finishing", "Pre-Finishing"]),
        ("TFT", True,  True,  False, True,  ["Finishing", "Pre-Finishing"]),
        ("TFT", True,  True,  False, True,  ["Finishing"]),
        ("TFT", True,  True,  False, True,  ["Pre-Finishing"]),
    ]


    for data_path, n_windows, base_path in datasets:
        for model_type, with_exo, with_sound, only_sound, with_optuna, passes in configs:
            print(f"{model_type} exo={with_exo} sound={with_sound} optuna={with_optuna} data={data_path} n_windows={n_windows} passes={passes}")
            train(
                model_type=model_type,
                data_path=data_path,
                with_exogenous_variables=with_exo,
                with_sound_columns=with_sound,
                only_sound_columns=only_sound,
                with_optuna=with_optuna,
                n_pre_process_window=n_windows,
                params=params,
                passes=passes,
                base_path=base_path
            )