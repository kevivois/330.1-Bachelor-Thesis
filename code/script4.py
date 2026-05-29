import matplotlib
matplotlib.use('Agg')
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import tsfel
from multiprocessing.pool import ThreadPool  
from tqdm import tqdm
import os


import tsfel

def get_custom_tsfel_cfg(fs=40000):
    return {
        'statistical': {
            'Min': {'parameters': '', 'function': 'tsfel.calc_min', 'use': 'yes'},
            'Max': {'parameters': '', 'function': 'tsfel.calc_max', 'use': 'yes'},
            'Mean': {'parameters': '', 'function': 'tsfel.calc_mean', 'use': 'yes'},
            'Median': {'parameters': '', 'function': 'tsfel.calc_median', 'use': 'yes'},
            'Standard deviation': {'parameters': '', 'function': 'tsfel.calc_std', 'use': 'yes'},
            'Kurtosis': {'parameters': '', 'function': 'tsfel.kurtosis', 'use': 'yes'},
            'Skewness': {'parameters': '', 'function': 'tsfel.skewness', 'use': 'yes'}
        },
        'temporal': {
            'Zero crossing rate': {'parameters': '', 'function': 'tsfel.zero_cross', 'use': 'yes'}
        },
        'spectral': {
            'Spectral centroid': {'parameters': {'fs': fs}, 'function': 'tsfel.spectral_centroid', 'use': 'yes'},
            'Spectral roll-off': {'parameters': {'fs': fs}, 'function': 'tsfel.spectral_roll_off', 'use': 'yes'}
        }
    }



def process(args):
        
        row, basepath, cols, points_per_sec = args
        
        sensor_file = row["sensor_file"]

        sensor_df = pl.scan_parquet(basepath + "/" + sensor_file)
        
        #cfg = tsfel.get_features_by_domain(["statistical","temporal","spectral"])

        summary = pl.from_pandas(tsfel.time_series_features_extractor(get_custom_tsfel_cfg(points_per_sec), sensor_df.select(cols).collect().to_pandas(), fs=points_per_sec, window_size=points_per_sec)).lazy().with_row_index("idx").with_columns(pl.lit(sensor_file).alias("sensor_file"))
        timestamp_summary = sensor_df.select(["timestamp","time"]).with_row_index("idx").with_columns((pl.col("idx")//points_per_sec).cast(pl.UInt32).alias("group")).group_by("group").agg(
            pl.col("timestamp").min().alias("timestamp"),
            pl.col("time").min().alias("time")
        )
        meta = pl.DataFrame([row]).lazy()

        joined  = summary.join(other=meta, on="sensor_file", how="left")
        joined = joined.join(other=timestamp_summary,how="cross")
        return joined


def create_csv_file(output_filename):
    basepath = "/run/media/kevivois/T7/BACHELOR/"
    filepath = basepath + "aggregated.parquet"

    points_per_sec = 40000

    aggregated = pl.scan_parquet(filepath)
    cols = ["AccX", "AccY", "AccZ", "Sound"]

    rows = aggregated.collect().to_dicts()
    
    tasks = [(row, basepath, cols, points_per_sec) for row in rows]
    dfs_extracted = []
    
    with ThreadPool(processes=os.cpu_count() - 2) as pool: 
        for chunk_df in tqdm(pool.imap_unordered(process, tasks), total=len(tasks), desc="Extraction TSFEL"):
            dfs_extracted.append(chunk_df)
    
    
    result = pl.concat(dfs_extracted)
    
    result.write_csv(f"{output_filename}")

    print(result)
    
    
    
def plot_csv():

    result = pl.scan_csv("test.csv")

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
        data["Sound_min"],
        c=color_codes,
        cmap="tab10",
        s=5,
        label="Sound Min"
    )
    axes[1].legend()

    axes[2].scatter(
        data["x"],
        data["Sound_max"],
        c=color_codes,
        cmap="tab10",
        s=5,
        label="Sound Max"
    )
    axes[2].legend()

    # Add colorbar for pass_type mapping
    sm = plt.cm.ScalarMappable(cmap="tab10", norm=plt.Normalize(vmin=0, vmax=len(unique_types)-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, label="Pass Type")
    cbar.set_ticks(range(len(unique_types)))
    cbar.set_ticklabels(unique_types)


    axes[0].set_ylim(-0.02, 0.02)
    axes[1].set_ylim(-5.0,1.0)
    axes[2].set_ylim(0, 5.0)


    data = data.with_columns(
        (pl.col("PassNumber") != pl.col("PassNumber").shift()).alias("change")
    )
    ''
    change_points = data.filter(pl.col("change"))["x"].to_list()
    for x in change_points:
            axes[0].axvline(x, color="black", alpha=0.3, linewidth=1)
        

    plt.show()
    
    
def main():
    create_csv_file("tsfel_extracted.csv")
    #print(tsfel.get_features_by_domain(["statistical","temporal","spectral"]))


if __name__ == "__main__":
    main()

