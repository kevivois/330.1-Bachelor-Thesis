import polars as pl
from .models.v2.LSTMModel import LSTMModel
from sklearn.model_selection import KFold
from pathlib import Path
import shutil
import json
import numpy as np
def train():
    data_path = "tsfel_extracted_v5.csv"
    passes = ["Finishing", "Pre-Finishing"]

    
    with_exogenous_variables = False
    with_sound_columns       = False
    only_sound_columns       = True

    meta_cols = [
        "sensor_file", "timestamp", "time", "ToolIdx", "plate_id",
        "DB_PASSES/NUMERO_OF", "PassID", "start_pos", "end_pos",
        "DB_PASSES/NUMERO_PASSE", "timestamp_right", "y", "pass_type",
        "index", "DB_PASSES/COMPTEUR_RESET_PASSES"
    ]
    sound_prefixes = ["Sound", "AccZ", "AccY", "AccX"]
    exogenous_variables = [
        "DB_PASSES/SELECTION_ALLIAGE", "DB_PASSES/EPAISSEUR_BRUTE",
        "DB_PASSES/PASSE_ACTIVE.EPAISSEUR", "Axe_X_master/ActualVelocity_Mean",
        "Broche/ActualSpeed_Mean", "pass_type"
    ]
    dummies_cols = [
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
                pl.col(c).shift(-1).over("sensor_file").alias(f"next_{c}")
            )
        data = data.drop_nulls(subset=[f"next_{c}" for c in exogenous_variables])

    
    existing_dummies = [c for c in dummies_cols if c in data.columns]
    data = data.to_dummies(columns=existing_dummies)

    
    feature_cols = [c for c in data.columns if c not in meta_cols]
    
    if only_sound_columns:
        feature_cols = [
            c for c in feature_cols
            if any(c.startswith(p) for p in sound_prefixes)
        ]
    elif not with_sound_columns:
        feature_cols = [
            c for c in feature_cols
            if not any(c.startswith(p) for p in sound_prefixes)
        ]

    cols_to_use = feature_cols + [c for c in meta_cols if c in data.columns]
    data = data.select(cols_to_use)
    
    kf = KFold(n_splits=4, shuffle=False)
    base_path = "runs"
    folder = Path(base_path + "/k-fold-run-"+LSTMModel.get_formated_datetime())
    folder.mkdir(exist_ok=True)
    
    
    tool_ids    = data["ToolIdx"].unique().sort().to_list()    
    cv_scores = {
        "rows":[],
        "summary":{}
    }
    d = kf.split(X=tool_ids)
    for fold_idx, (train_val_idx, test_idx) in enumerate(d):
        test_tools  = [tool_ids[i] for i in test_idx]
        train_val   = [tool_ids[i] for i in train_val_idx]
        val_tools   = train_val[-2:]
        train_tools = train_val[:-2]

        print(train_tools,val_tools,test_tools)
        
        params =  {
        "model": "LSTM",
        "input_chunk_length": 40,
        "output_chunk_length": 1,
        "hidden_dim": 48,
        "n_rnn_layers": 1,
        "batch_size": 64,
        "dropout": 0.2,
        "n_epochs": 50,
        "learning_rate": 0.004845230559659738
        }
        
        model = LSTMModel(data.lazy(),**params,train_tools=train_tools,test_tools=test_tools,val_tools=val_tools,filepath=data_path,meta_cols=meta_cols,n_epochs=50,base_image_path=base_path)
        
        
        #best_params = model.optimize_parameters(n_trials=10,n_epochs_optuna=25)
        #print(f"best params: {best_params}")

        path = model.start_training()
        
        cv_scores["rows"].append({
            "fold":        fold_idx + 1,
            "r2":          model.res_r2,
            "rmse":        model.res_rmse,
            "mae":         model.res_mae,
            "train_tools": train_tools,
            "val_tools":   val_tools,
            "test_tools":  test_tools,
            "path":str(path)
        })
        
    r2_list = [row["r2"] for row in cv_scores["rows"]]
    rmse_list = [row["rmse"] for row in cv_scores["rows"]]
    mae_list = [row["mae"] for row in cv_scores["rows"]]
    
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
    json_string = json.dumps(cv_scores, indent=4)
    result_file.write_text(json_string)

        
        
    
    


        
    
if __name__ == "__main__":
    train()