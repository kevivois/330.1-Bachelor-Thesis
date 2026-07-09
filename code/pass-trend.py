# This script plots the selected target variable over time for each tool, colors the points by pass type, and adds a global linear trend line

import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def plot_passes_trend(filename: str, y_col: str = "Broche/StatusTorqueData.ActualTorque_Mean"):
    

    df: pl.DataFrame = pl.scan_csv(filename).collect()
    df_fp = df.sort(["ToolIdx", "timestamp"])
    tool_ids = df_fp["ToolIdx"].unique().sort().to_list()
    color_map = {"Finishing": "tab:blue", "Pre-Finishing": "tab:orange"}

    for tool in tool_ids:
        data = df_fp.filter((pl.col("ToolIdx") == tool) & (pl.col("pass_type").is_in(["Pre-Finishing","Finishing"]))).with_row_index("x")
        
        # Map pass types to colors
        colors = data.get_column("pass_type").to_numpy()
        c = [color_map.get(p, "gray") for p in colors]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.scatter(data.get_column("x").to_numpy(), data.get_column(y_col).to_numpy(), c=c, s=4)

         # Mark pass boundaries
        change_points = (
            data.filter(pl.col("PassID") != pl.col("PassID").shift(1)).get_column("x").to_list()
        )
        for cp in change_points:
            ax.axvline(cp, color="black", alpha=0.15, linewidth=0.8)

        ax.set_title(f"Outil {tool} Couple - Orange : Finishing , Blue : Pre-Finishing")
        ax.set_ylabel("Couple (Nm)")
        ax.set_xlabel("index")
        ax.grid(alpha=0.5)

        # Fit and plot a linear trend line
        z = np.polyfit(data.get_column("x").to_numpy(), data.get_column(y_col).to_numpy(), 1)
        trend = np.poly1d(z)
        ax.plot(data.get_column("x"), trend(data.get_column("x").to_numpy()), color="red", linewidth=2, linestyle="--",
                label=f"trend")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return df_fp


if __name__ == "__main__":
    plot_passes_trend( "../data/v5/tsfel_extracted_v5.csv")