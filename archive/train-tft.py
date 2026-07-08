import polars as pl
from TFTModel import TFTModel

def train():
    data_path = "tsfel_extracted_v5.csv"
    passes = ["Finishing", "Pre-Finishing"]

    
    with_exogenous_variables = True
    with_sound_columns       = True
    only_sound_columns       = False

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

    tool_ids    = data["ToolIdx"].unique().sort().to_list()
    train_tools = tool_ids[:-4]
    val_tools   = tool_ids[-4:-2]
    test_tools  = tool_ids[-2:]
    
    print(train_tools,val_tools,test_tools)
    
    model = TFTModel(data.lazy(),train_tools=train_tools,test_tools=test_tools,val_tools=val_tools,filepath=data_path,meta_cols=meta_cols,n_epochs=50,base_image_path="./runs")
    
    best_params = model.optimize_parameters(n_trials=10,n_epochs_optuna=25)
    print(f"best params: {best_params}")
    
    path = model.train()
    print(path)
    
if __name__ == "__main__":
    train()