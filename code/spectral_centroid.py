# This script reads the extracted CSV dataset, filters it by selected pass types, and plots the evolution of the sound spectral centroid for each tool.

import matplotlib
matplotlib.use('TkAgg')
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def main():
    filename = "../data/v4/tsfel_extracted_new.csv"
    passes = ["Pre-Finishing", "Finishing"]
    data_pl = pl.scan_csv(filename)
    toolIdx = data_pl.select("ToolIdx").unique().collect().sort("ToolIdx")["ToolIdx"].to_list()
    for toolIdx in  toolIdx:
        
        data = data_pl.filter(
            (pl.col("ToolIdx") == toolIdx) & 
            (pl.col("pass_type").is_in(passes))
        ).sort("timestamp")
        
        result = data.collect()
        
        if len(result) == 0:
            continue
        print("creating for figure outil - ",toolIdx)
        plt.figure(figsize=(15, 10), dpi=200)
        
        colors = {"Pre-Finishing": "#1f77b4", "Finishing": "#ff7f0e"}
        x_axis = np.arange(len(result))
        signal_name = "Sound_Spectral centroid"
        y_signal = result[signal_name].to_numpy()
        
        for p_type in passes:
            mask = result["pass_type"].to_numpy() == p_type
            if np.any(mask):
                plt.scatter(
                    x_axis[mask],
                    y_signal[mask],
                    label=p_type,
                    color=colors[p_type],
                    s=20,
                    alpha=0.6,
                    edgecolors='none'
                )
        plt.title(f"Sound Spectral centroid evolution : Outil {toolIdx}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("index", fontsize=11, labelpad=10)
        plt.ylabel("spectral centroid HZ", fontsize=11, labelpad=10)
        
        plt.grid(True, linestyle=":", alpha=0.6, color="#bdc3c7")
        plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ebdcdc", framealpha=0.9, fontsize=10)
        plt.tight_layout()
        #plt.savefig(f"images/v2/stats/spectral_centroid_outil_{toolIdx}.png", dpi=200)
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()