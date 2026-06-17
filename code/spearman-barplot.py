import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
from scipy.stats import spearmanr

def compute_spearman_top10(data: pl.DataFrame):

    meta_cols = [
        "pass_type", "sensor_file", "DB_PASSES/NUMERO_OF", 
        "DB_PASSES/SELECTION_ALLIAGE", "DB_PASSES/COMPTEUR_RESET_PASSES","DB_PASSES_NUMERO_PASSE", 
        "PassNumber", "ToolIdx", "PassID", "y", "timestamp", "time", "index"
    ]
    str_parts = "Broche/StatusTorqueData"
    feature_cols = [c for c in data.columns if c not in meta_cols and str_parts not in c]
    
    if not feature_cols:
        return None

    y_np = data["y"].to_numpy()
    correlations = []
    
    for col in feature_cols:
        feat_np = data[col].to_numpy()
        corr, _ = spearmanr(feat_np, y_np)
        if not np.isnan(corr):
            correlations.append((col, abs(corr)))
            
    correlations.sort(key=lambda x: x[1], reverse=True)
    return correlations[:10]
    
def plot(filename=""):
    os.makedirs("images/correlations", exist_ok=True)
    result = pl.scan_csv(filename)
    
    tools = result.select(pl.col("ToolIdx").unique()).collect().get_column("ToolIdx").to_list()
    tools = [t for t in tools if t is not None]
    tools.sort()
    passes = ["Pre-Finishing","Finishing"]
    pass_types = result.select(pl.col("pass_type")).filter(pl.col("pass_type").is_in(passes)).unique().collect().get_column("pass_type").to_list()
    pass_types = [p for p in pass_types if p is not None]
    
    for p in pass_types:
        all_tool_corrs = {}
        
        for t in tools:
            data = result.filter((pl.col("ToolIdx") == t) & (pl.col("pass_type") == p)).collect()
            top_corrs = compute_spearman_top10(data)
            if top_corrs:
                all_tool_corrs[t] = top_corrs
                    
        if all_tool_corrs:
            num_tools = len(all_tool_corrs)
            cols = 4
            rows = int(np.ceil(num_tools / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=(22, 5 * rows), squeeze=False)
            axes = axes.flatten()
            
            for idx, (t_idx, corrs) in enumerate(all_tool_corrs.items()):
                names = [c[0] for c in corrs]
                values = [c[1] for c in corrs]
                
                ax = axes[idx]
                ax.barh(names, values, color=plt.cm.plasma(np.linspace(0.8, 0.4, len(values))))
                ax.set_title(f"Outil {t_idx}", fontsize=12, fontweight='bold')
                ax.set_xlim(0, 1.0)
                ax.grid(True, axis='x', linestyle='--', alpha=0.5)
                ax.invert_yaxis()
                ax.tick_params(axis='y', labelsize=8)
                
            for j in range(num_tools, len(axes)):
                fig.delaxes(axes[j])
                
            safe_p_name = p.replace(" ", "_").replace("/", "_")
            fig.suptitle(f"Corrélation de Spearman — Régime : {p}", fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(f"images/v2/stats/spearman plot/spearman_{safe_p_name}.png", dpi=200)
            #plt.show()
            plt.close(fig)

def main():
    plot("data/v4/tsfel_extracted_new.csv")

if __name__ == "__main__":
    main()