import matplotlib
matplotlib.use('TkAgg')
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tsfel
from concurrent.futures import ProcessPoolExecutor  
from tqdm import tqdm
import os
import json
from tsfresh import select_features


def process(args):
    row, basepath, cols, points_per_sec, row_id = args
    print("processing ", row_id, flush=True)
    sensor_file = row["sensor_file"]
    
    sensor_df = pl.scan_parquet(basepath + "/" + sensor_file)
    data_pl = sensor_df.select(cols).collect()
    sensor_length = len(data_pl)
    points_per_sec = sensor_length
    print(points_per_sec)
    signal = data_pl.to_numpy()
    
    cfg = tsfel.get_features_by_domain()
    
    features_whitelist = {
        'temporal': ['Peak to peak distance', 'Zero crossing rate'],
        'statistical': ['Mean', 'Standard deviation', 'Max', 'Min', 'Variance', 'Root mean square'],
        'spectral': ['Spectral centroid', 'Spectral roll-off'],
        #'fractal': ['Hurst exponent']
        'fractal': ['']
    }
    
    for domain, allowed_features in features_whitelist.items():
        if domain in cfg:
            cfg[domain] = {k: v for k, v in cfg[domain].items() if k in allowed_features}
            for feature_name in cfg[domain]:
                cfg[domain][feature_name]['use'] = 'yes'

    
    extracted_features_pd = tsfel.time_series_features_extractor(
        cfg, 
        signal, 
        fs=points_per_sec, 
        window_size=points_per_sec, 
        verbose=0,
        n_jobs=max(1,os.process_cpu_count()-2)
    )
    
    extracted_features_pd['id'] = np.arange(len(extracted_features_pd), dtype=np.int64)

    summary = pl.from_pandas(extracted_features_pd).lazy()
    summary = summary.with_columns(pl.lit(sensor_file).alias("sensor_file"))
    
    
    n_windows = len(extracted_features_pd)
    timestamp_summary = sensor_df.select(["timestamp", "time"]).collect()
    timestamp_summary = timestamp_summary.gather_every(points_per_sec).head(n_windows).with_row_index("id")
    
    meta = pl.DataFrame([row])
    joined = summary.collect().join(other=meta, on="sensor_file", how="left")
    joined = joined.join(other=timestamp_summary, on="id", how="left")
    return joined.lazy()


def create_csv_file(output_filename, target="Broche/StatusTorqueData.ActualTorque"):
    basepath = "/home/kevin.voisin/datasets/kevivois/"
    filepath = basepath + "aggregated.parquet"

    points_per_sec = 40000

    aggregated = pl.scan_parquet(filepath)
    cols = ["AccX", "AccY", "AccZ", "Sound"]
    
    aggregated = aggregated.sort("sensor_file").with_columns(
        pl.col(target).shift(-1).alias("y")
    )

    rows = aggregated.collect().to_dicts()
    
    tasks = [(row, basepath, cols, points_per_sec, row_id) for row_id, row in enumerate(rows)]
    dfs_extracted = []
    
    print("starting processing", flush=True)
    with ProcessPoolExecutor(max_workers=max(1,os.process_cpu_count()-2)) as pool: 
        for chunk_df in tqdm(pool.map(process, tasks), total=len(tasks), desc="Extraction TSFEL", unit="file"):
            dfs_extracted.append(chunk_df)
    
    result = pl.concat(dfs_extracted).collect()
    result.write_csv(f"{output_filename}_total.csv")
    result = result.drop_nulls(subset=["y"])
    
    result = result.drop("id").with_row_index("id")
    
    
    custom_meta = [c for c in result.columns if c in rows[0].keys() and c != "y"]
    meta_cols = ["id", "sensor_file", "timestamp", "time", "y"] + custom_meta
    meta_cols = list(set(meta_cols))
    import polars.selectors as cs
    x_pd = result.drop(meta_cols).select(cs.numeric()).to_pandas().astype(np.float64)
    x_pd.index = result.get_column("id").to_numpy()
    
    y_series = result.get_column("y").to_pandas()
    y_series.index = result.get_column("id").to_numpy()
    
    selected_features = select_features(X=x_pd, y=y_series, n_jobs=max(1, os.process_cpu_count() - 2))
    
    features_clean = pl.from_pandas(selected_features.reset_index()).rename({"index": "id"})
    meta_clean = result.select(meta_cols)
    
    output = features_clean.join(meta_clean, on="id", how="inner")
    output.write_csv(f"{output_filename}")

    
    
def plot_csv(filename):
    result = pl.scan_csv(filename)
    data = result.filter(pl.col("ToolIdx") == 2).with_row_index("x").collect()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    pass_types = data["pass_type"].cast(pl.Categorical).to_numpy()
    unique_types = sorted(set(pass_types))
    type_to_code = {ptype: i for i, ptype in enumerate(unique_types)}

    color_codes = np.array([type_to_code[ptype] for ptype in pass_types])

    axes[0].scatter(
        data["x"],
        data["Sound_mean"],
        c=color_codes,
        cmap="tab10",
        s=5,
        label="Sound Feature 1"
    )
    axes[0].legend()

    axes[1].scatter(
        data["x"],
        data["Sound__mean"],
        c=color_codes,
        cmap="tab10",
        s=5,
        label="Sound Feature 2"
    )
    axes[1].legend()

    axes[2].scatter(
        data["x"],
        data["Sound__maximum"],
        c=color_codes,
        cmap="tab10",
        s=5,
        label="Sound Feature 3"
    )
    axes[2].legend()

    sm = plt.cm.ScalarMappable(cmap="tab10", norm=plt.Normalize(vmin=0, vmax=len(unique_types)-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, label="Pass Type")
    cbar.set_ticks(range(len(unique_types)))
    cbar.set_ticklabels(unique_types)

    axes[0].set_ylim(-0.02, 0.02)
    axes[1].set_ylim(-5.0, 1.0)
    axes[2].set_ylim(0, 5.0)

    data = data.with_columns(
        (pl.col("PassNumber") != pl.col("PassNumber").shift(1)).alias("change")
    )
    
    change_points = data.filter(pl.col("change"))["x"].to_list()
    for x in change_points:
        axes[0].axvline(x, color="black", alpha=0.3, linewidth=1)
        
    plt.show()
    
    
def main():
    #create_csv_file("tsfel_extracted.csv", "Broche/StatusTorqueData.ActualTorque")
    create_csv_file("tsfel_extracted_new")

if __name__ == "__main__":
    main()