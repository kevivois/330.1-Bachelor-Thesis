import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools

def plot_csv_3d(data:pl.DataFrame):
    if len(data) == 0:
        return  None
    
    target_col = "Broche/StatusTorqueData.ActualTorque"
    data = data.with_columns(pl.col(target_col).shift(-1).alias("y")).drop_nulls(subset=["y"])

    meta_cols = ["id","pass_type","DB_PASSES/SELECTION_ALLIAGE" "sensor_file", "timestamp", "time", "ToolIdx", "PassNumber", "y", "plate_id", "DB_PASSES/NUMERO_OF", "PassID", "start_pos", "end_pos", "DB_PASSES/NUMERO_PASSE", "Broche/StatusTorqueData.ActualTorque", "timestamp_right"]
    feature_cols = [c for c in data.columns if c not in meta_cols]
    X = data.select(feature_cols).to_numpy()
    correlations = []
    for col in feature_cols:
        feat_np = data[col].to_numpy()
        y_np = data["y"].to_numpy()
        
        corr = np.corrcoef(np.argsort(feat_np), np.argsort(y_np))[0, 1]
        correlations.append((col, abs(corr)))
    correlations.sort(key=lambda x: x[1], reverse=True)

    return correlations[:10]
    
def plot(filename=""):
    result = pl.scan_csv(filename)
    tools = result.select(pl.col("ToolIdx")).unique().collect()["ToolIdx"].to_list()
    tools.sort()
    pass_types = result.select(pl.col("pass_type")).unique().collect()["pass_type"].to_list()
    
    for p in pass_types:
        all_tool_corrs = {}
        
        for t in tools:
            data = result.filter((pl.col("ToolIdx") == t) & (pl.col("pass_type") == p)).collect()
            top_corrs = plot_csv_3d(data)
            if top_corrs:
                all_tool_corrs[t] = top_corrs
                    
        if all_tool_corrs:
            num_tools = len(all_tool_corrs)
            cols = 4
            rows = int(np.ceil(num_tools / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), sharex=False)
            axes = axes.flatten()
            
            for idx, (t_idx, corrs) in enumerate(all_tool_corrs.items()):
                names = [c[0] for c in corrs]
                values = [c[1] for c in corrs]
                
                ax = axes[idx]
                bars = ax.barh(names, values, color=plt.cm.viridis(np.linspace(0.8, 0.3, len(values))))
                ax.set_title(f"Outil {t_idx}", fontsize=11, fontweight='bold')
                ax.set_xlim(0, 1.0)
                ax.grid(True, axis='x', linestyle='--', alpha=0.5)
                ax.invert_yaxis()
                ax.tick_params(axis='y', labelsize=9)
                
            for j in range(num_tools, len(axes)):
                fig.delaxes(axes[j])
                
            fig.suptitle(f"Correlation de spearman - : {p}", fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.show()
    

def main():
    plot("data/v1/tsfel_extracted.csv_total.csv")

if __name__ == "__main__":
    main()