import matplotlib
matplotlib.use('TkAgg')
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import tsfel
from multiprocessing.pool import ThreadPool  
from tqdm import tqdm
import os


from tsfresh import extract_features,select_features
from tsfresh.feature_extraction import EfficientFCParameters




def process(args):
    row, basepath, cols, points_per_sec = args
    sensor_file = row["sensor_file"]

    sensor_df = pl.scan_parquet(basepath + "/" + sensor_file)
    data_pd = sensor_df.select(cols).collect().to_pandas()
    
    data_pd['id'] = np.arange(len(data_pd)) // points_per_sec
    data_pd['time_idx'] = np.arange(len(data_pd)) % points_per_sec
    extracted_features_pd = extract_features(
        data_pd, 
        column_id='id', 
        column_sort='time_idx',
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=1, 
        disable_progressbar=True
    )

    summary = pl.from_pandas(extracted_features_pd.reset_index()).lazy()
    summary = summary.with_columns(pl.lit(sensor_file).alias("sensor_file"))
    
    
    
    timestamp_summary = sensor_df.select(["timestamp","time"]).with_row_index("idx").with_columns(
        (pl.col("idx") // points_per_sec).cast(pl.Int64).alias("index")
    ).group_by("index").agg(
        pl.col("timestamp").min().alias("timestamp"),
        pl.col("time").min().alias("time"),
    )
    
    meta = pl.DataFrame([row]).lazy()
    joined = summary.join(other=meta, on="sensor_file", how="left")
    joined = joined.join(other=timestamp_summary, on="index", how="left")
    return joined


def create_csv_file(output_filename,target="cutting_depth"):
    basepath = "/run/media/kevivois/T7/BACHELOR/"
    filepath = basepath + "aggregated.parquet"

    points_per_sec = 40000

    aggregated = pl.scan_parquet(filepath)
    cols = ["AccX", "AccY", "AccZ", "Sound"]
    
    aggregated = aggregated.sort("sensor_file").with_columns(
        pl.col(target).shift(-1).alias("y")
    )


    rows = aggregated.collect().to_dicts()
    
    tasks = [(row, basepath, cols, points_per_sec) for row in rows]
    dfs_extracted = []
    
    with ThreadPool(processes=os.cpu_count() - 2) as pool: 
        for chunk_df in tqdm(pool.imap_unordered(process, tasks), total=len(tasks), desc="Extraction TSFEL",unit="file"):
            dfs_extracted.append(chunk_df)
    
    
    result = pl.concat(dfs_extracted).lazy()
    target_col = result.collect().get_column("y")
    meta_cols = ["sensor_file", "timestamp", "time", "y"]
    x = result.drop(meta_cols).rename({"id":"index"})
    selected_features = select_features(X=x.collect().to_pandas(),y=target_col.to_pandas())
    
    output = pl.concat([pl.from_pandas(selected_features),result.select(meta_cols)],how="horizontal").collect()
    
    
    output.write_csv(f"{output_filename}")

    print(output)
    
    
    
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
        label="Sound Mean"
    )
    axes[0].legend()

    axes[1].scatter(
        data["x"],
        data["Sound__mean"],
        c=color_codes,
        cmap="tab10",
        s=5,
        label="Sound Minimum"
    )
    axes[1].legend()

    axes[2].scatter(
        data["x"],
        data["Sound__maximum"],
        c=color_codes,
        cmap="tab10",
        s=5,
        label="Sound Max"
    )
    axes[2].legend()

    sm = plt.cm.ScalarMappable(cmap="tab10", norm=plt.Normalize(vmin=0, vmax=len(unique_types)-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, label="Pass Type")
    cbar.set_ticks(range(len(unique_types)))
    cbar.set_ticklabels(unique_types)


    axes[0].set_ylim(-0.02, 0.02)
    axes[1].set_ylim(-5.0,1.0)
    axes[2].set_ylim(0, 5.0)


    data = data.with_columns(
        (pl.col("PassNumber") != pl.col("PassNumber").shift(1)).alias("change")
    )
    ''
    change_points = data.filter(pl.col("change"))["x"].to_list()
    for x in change_points:
            axes[0].axvline(x, color="black", alpha=0.3, linewidth=1)
        

    plt.show()
    
    
def main():
    #plot_csv("test.csv")
    create_csv_file("tsfel_extracted.csv","cutting_depth")

if __name__ == "__main__":
    main()

