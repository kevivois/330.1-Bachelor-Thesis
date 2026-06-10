import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools

def compute_pca_3d(data):
    data = data.collect()
    if len(data) == 0:
        return None, None, None
    
    target_col = "Broche/StatusTorqueData.ActualTorque"
    data = data.with_columns(pl.col(target_col).shift(-1).alias("y")).drop_nulls(subset=["y"])

    meta_cols = ["id", "sensor_file", "timestamp", "time", "ToolIdx", "PassNumber", "pass_type", "y", "plate_id", "DB_PASSES/NUMERO_OF","DB_PASSES/SELECTION_ALLIAGE" ,"PassID", "start_pos", "end_pos", "DB_PASSES/NUMERO_PASSE", "Broche/StatusTorqueData.ActualTorque", "timestamp_right"]
    feature_cols = [c for c in data.columns if c not in meta_cols]
    
    if len(data) < 5:
        return None, None, None
    
    X = data.select(feature_cols).to_numpy()
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=3)  
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    
    return X_pca, var_exp, np.arange(len(data))
    
def plot(filename=""):
    os.makedirs("images", exist_ok=True)
    
    result = pl.scan_csv(filename)
    tools = result.select(pl.col("ToolIdx")).unique().collect()["ToolIdx"].to_list()
    tools.sort()
    pass_types = result.select(pl.col("pass_type")).unique().collect()["pass_type"].to_list()
    
    for p, t in itertools.product(pass_types, tools):
        data = result.filter((pl.col("ToolIdx") == t) & (pl.col("pass_type") == p))
        X_pca, var_exp, indices = compute_pca_3d(data)
        
        if X_pca is not None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            sc = ax.scatter(
                X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                c=indices, 
                cmap="viridis", 
                s=25, 
                alpha=0.8,
                edgecolors='w',
                linewidths=0.2
            )
            
            ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], color='black', alpha=0.15, linewidth=1)
            
            var_totale = np.sum(var_exp) * 100
            ax.set_title(f"Espace Latent PCA 3D - Outil {t}\nRégime : {p} (Variance Totale Expliquée : {var_totale:.1f}%)", fontsize=12, fontweight='bold')
            ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
            ax.set_zlabel(f"PC3 ({var_exp[2]*100:.1f}%)")
            
            cbar = fig.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label("Progression temporelle dans le cycle (Secondes)")
            
            plt.tight_layout()
            
            safe_p_name = p.replace(" ", "_").replace("/", "_")
            fig_name = f"images/PCA_Outil_{t}_{safe_p_name}.png"
            #plt.savefig(fig_name, dpi=150)
            plt.show()
            plt.close(fig)

def main():
    plot("data/v1/tsfel_extracted.csv_total.csv")

if __name__ == "__main__":
    main()