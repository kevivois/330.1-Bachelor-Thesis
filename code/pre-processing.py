from dataclasses import dataclass

import matplotlib
matplotlib.use('TkAgg')
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import tsfel
from concurrent.futures import ProcessPoolExecutor  
from tqdm import tqdm
import os
import polars.selectors as cs
from tsfresh import select_features
import pathlib

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

WINDOW_SENSOR_THRESHOLD = 50
WINDOW_TORQUE_THRESHOLD = 5
WINDOW_RATIO_THRESHOLD = 10

@dataclass(kw_only=True)
class TaskParams:
    filename:str
    basepath:pathlib.Path
    sensor_cols:list[str]
    torque_cols:list[str]
    sensor_frequency:int
    torque_frequency:int
    n_windows:int
    cfg:dict

def process(args:TaskParams):
    
    sensor_df = pl.scan_parquet(args.basepath / "passes_sensor" / args.filename)
    torque_df = pl.scan_parquet(args.basepath / "passes" / args.filename)
    
    
    try:
        sensor_schema = sensor_df.collect_schema()
        torque_schema = torque_df.collect_schema()
    except Exception:
        return None

    signal_sensor = sensor_df.select(args.sensor_cols).collect().to_numpy()
    window_sensor_size = len(signal_sensor) // args.n_windows
    print(window_sensor_size,"window_sensor_size")
    if window_sensor_size == 0 or window_sensor_size < WINDOW_SENSOR_THRESHOLD:
        return None


    print(f"starting feature extraction for sensor {args.filename}")
    features_sensor_pd = tsfel.time_series_features_extractor(
        args.cfg, signal_sensor, fs=args.sensor_frequency, window_size=window_sensor_size, verbose=0, n_jobs=1
    )
    print(f"finished feature extraction for sensor {args.filename}")
    
    sensor_summary = pl.from_pandas(features_sensor_pd).with_row_index("index")
    n_windows_sensor = len(sensor_summary)
    
    sensor_meta_cols = [c for c in sensor_schema.keys() if c not in args.sensor_cols]
    
    sensor_rename = {
    col: col.replace(f"{i}_", f"{args.sensor_cols[i]}_")
    for i in range(len(args.sensor_cols))
    for col in sensor_summary.columns
    if col.startswith(f"{i}_")
    }
    sensor_summary = sensor_summary.rename(sensor_rename)
    
    sensor_meta = (
        sensor_df.select(sensor_meta_cols)
        .gather_every(window_sensor_size)
        .head(n_windows_sensor)
        .with_row_index("index")
    ).collect()
    
    
    signal_torque = torque_df.select(args.torque_cols).collect().to_numpy()
    window_torque_size = len(signal_torque) // args.n_windows
    
    if window_torque_size == 0 or window_torque_size < WINDOW_TORQUE_THRESHOLD:
        print("returning None")
        return None
    
    print(f"starting feature extraction for torque{args.filename}")
    features_torque_pd = tsfel.time_series_features_extractor(
        args.cfg,signal_torque,fs=args.torque_frequency,window_size=window_torque_size,verbose=0,n_jobs=1
    )
    print(f"finished feature extraction for torque {args.filename}")
    torque_summary = pl.from_pandas(features_torque_pd).with_row_index("index")
    n_windows_torque = len(torque_summary)
    
    torque_rename = {
    col: col.replace(f"{i}_", f"{args.torque_cols[i]}_")
    for i in range(len(args.torque_cols))
    for col in torque_summary.columns
    if col.startswith(f"{i}_")
    }
    torque_summary = torque_summary.rename(torque_rename)
    
    torque_meta_cols_raw = [c for c in torque_schema.keys() if c not in args.torque_cols]
    dup_cols = [c for c in sensor_meta_cols if c in torque_meta_cols_raw]
    torque_meta_cols = [c for c in torque_meta_cols_raw if c not in dup_cols]
    
    torque_meta = (
        torque_df.select(torque_meta_cols)
        .gather_every(window_torque_size)
        .head(n_windows_torque)
        .with_row_index("index")
    ).collect()
    

    torque_features_df = torque_summary.join(torque_meta,on="index",how="inner")
    
    sensor_features_df = sensor_summary.join(sensor_meta, on="index", how="inner")
    
    final_df = sensor_features_df.join(torque_features_df,on="index",how="inner")
    
    return final_df.with_columns(pl.lit(args.filename).alias("sensor_file"))
    
    
def create_csv_file(output_filename, target="Broche/StatusTorqueData.ActualTorque"):
    basepath = pathlib.Path("/home/kevin.voisin/datasets/kevivois/data_test")
    window_size = WINDOW_RATIO_THRESHOLD
    sensor_frequency = 40000
    torque_frequency = 10
    sensor_cols = ["AccX", "AccY", "AccZ", "Sound"]
    torque_cols = ["Broche/ActualSpeed", "Axe_X_master/ActualVelocity", "Broche/StatusTorqueData.ActualTorque", "Axe_X_master/ActualPosition"]
    rows = [f.name for f in (basepath / "passes_sensor").iterdir() if f.is_file() and f.suffix == ".parquet"]

    features = {
        'temporal': ['Peak to peak distance', 'Zero crossing rate'],
        'statistical': ['Mean', 'Standard deviation', 'Max', 'Min', 'Variance', 'Root mean square'],
        'spectral': ['Spectral centroid', 'Spectral roll-off'],
        'fractal': []
    }
    
    '''features = {
        'temporal': [],
        'statistical':['Mean'],
        'spectral': [],
        'fractal': []
    }'''
    
    
    cfg = tsfel.get_features_by_domain()
    for domain, feature_list in features.items():
        if domain in cfg:
            cfg[domain] = {k: v for k, v in cfg[domain].items() if k in feature_list}
            for feature_name in cfg[domain]:
                cfg[domain][feature_name]['use'] = 'yes'
    
    tasks = [TaskParams(filename=filename, basepath=basepath, sensor_cols=sensor_cols, torque_cols=torque_cols, sensor_frequency=sensor_frequency, torque_frequency=torque_frequency, n_windows=window_size,cfg=cfg) for filename in rows]
    dfs_extracted = []
    
    print(f"Starting processing of {len(rows)} files...", flush=True)
    workers = max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=workers) as pool: 
        for chunk_df in tqdm(pool.map(process, tasks), total=len(tasks), desc="Extraction TSFEL", unit="file"):
            if chunk_df is not None:
                dfs_extracted.append(chunk_df)
    
    if not dfs_extracted:
        print("Aucune donnée n'a été extraite.")
        return

    result = (pl.DataFrame(pl.concat(dfs_extracted, how="diagonal"))).sort(["sensor_file","timestamp"]).drop("index").with_row_index("index")
    real_target = f"{target}_Mean"
    
    result = result.with_columns(pl.col(real_target).shift(-1).alias("y")).drop_nulls(subset=["y"])
    
    v = pl.col("Axe_X_master/ActualVelocity_Mean")
    
    result = result.with_columns(
        pl.when(v.is_null() | (v == 0)).then(pl.lit("Unknown"))
            .when((v > 0) & (v < 900)).then(pl.lit("Finishing"))
            .when((v >= 1100) & (v < 1300)).then(pl.lit("Pre-Finishing"))
            .when((v >= 1300) & (v < 1500)).then(pl.lit("Blanking"))
            .when((v >= 2300) & (v < 2500)).then(pl.lit("Roughing"))
            .otherwise(pl.lit("Unknown"))
            .alias("pass_type")
    )
    
    result.write_csv(f"{output_filename}_complete_file.csv")
    print(f"successfully written f{output_filename}_complete_file.csv")

    string_categorical_columns = ["pass_type","sensor_file","DB_PASSES/NUMERO_OF","DB_PASSES/SELECTION_ALLIAGE","DB_PASSES/NUMERO_OF","DB_PASSES/COMPTEUR_RESET_PASSES","PassNumber","ToolIdx","PassID","y","timestamp","time","index"]
    X = result.drop(string_categorical_columns).to_pandas().astype(np.float64)
    X.index = result.get_column("index").to_numpy()
    Y = result.select("y")["y"].to_numpy()
    
    meta_values = result.select(string_categorical_columns)
    
    selected_features = select_features(X=X,y=Y,n_jobs=workers)
    
    output = pl.from_pandas(selected_features)
    output = output.with_columns(pl.lit(selected_features.index.values).alias("index"))
    output = output.join(meta_values,on="index",how="inner")
    
    output.write_csv(f"{output_filename}.csv")
    print(f"successfully written f{output_filename}.csv")
    
  


def plot_csv(filename):
    result = pl.scan_csv(filename)
    data = result.filter(pl.col("ToolIdx") == 2).with_row_index("x").collect()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    pass_types = data["pass_type"].cast(pl.Categorical).to_numpy()
    unique_types = sorted(set(pass_types))
    type_to_code = {ptype: i for i, ptype in enumerate(unique_types)}
    color_codes = np.array([type_to_code[ptype] for ptype in pass_types])

    axes[0].scatter(data["x"], data["Sound_mean"], c=color_codes, cmap="tab10", s=5, label="Sound Feature 1")
    axes[0].legend()
    axes[1].scatter(data["x"], data["Sound__mean"], c=color_codes, cmap="tab10", s=5, label="Sound Feature 2")
    axes[1].legend()
    axes[2].scatter(data["x"], data["Sound__maximum"], c=color_codes, cmap="tab10", s=5, label="Sound Feature 3")
    axes[2].legend()

    sm = plt.cm.ScalarMappable(cmap="tab10", norm=plt.Normalize(vmin=0, vmax=len(unique_types)-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, label="Pass Type")
    cbar.set_ticks(range(len(unique_types)))
    cbar.set_ticklabels(unique_types)

    axes[0].set_ylim(-0.02, 0.02)
    axes[1].set_ylim(-5.0, 1.0)
    axes[2].set_ylim(0, 5.0)

    data = data.with_columns((pl.col("PassNumber") != pl.col("PassNumber").shift(1)).alias("change"))
    change_points = data.filter(pl.col("change"))["x"].to_list()
    for x in change_points:
        axes[0].axvline(x, color="black", alpha=0.3, linewidth=1)
        
    plt.show()
    
def main():
    create_csv_file("tsfel_extracted_new")

if __name__ == "__main__":
    main()