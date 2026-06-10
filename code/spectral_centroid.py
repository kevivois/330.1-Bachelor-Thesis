import matplotlib
matplotlib.use('TkAgg')
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import tsfel
from concurrent.futures import ProcessPoolExecutor  
from tqdm import tqdm
import os



def main():
    filename = "data/v1/tsfel_extracted.csv"
    toolIdx = 1
    data = pl.scan_csv(filename).filter(pl.col("ToolIdx") == toolIdx)
    result = data.collect()

    pass_types = result["pass_type"].to_numpy()

    unique_types, color_ids = np.unique(pass_types, return_inverse=True)

    plt.scatter(
        np.arange(len(result)),
        result["3_Spectral centroid"],
        c=color_ids,
        cmap="tab10",  
        s=5
    )

    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(len(unique_types)))
    cbar.set_ticklabels(unique_types)

    plt.savefig("spectral_centroid.png")




if __name__ == "__main__":
    main()