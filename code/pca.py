# This script computes and visualizes 3D PCA projections for each tool and pass type.


import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

'''
Function used to generate/show & save in a row a PCA by ToolIdx and by type of passes
'''
def compute_pca_3d(data:pl.LazyFrame):
    data = data.collect()
    if len(data) == 0:
        return None, None, None
    
    # columns to exclude from features
    meta_cols = ["id", "sensor_file", "timestamp", "time", "ToolIdx", "PassNumber", "pass_type","y", "DB_PASSES/NUMERO_OF","DB_PASSES/SELECTION_ALLIAGE" ,"PassID", "start_pos", "end_pos", "DB_PASSES/NUMERO_PASSE", "timestamp_right"]
    feature_cols = [c for c in data.columns if c not in meta_cols]
    
    
    # minimum sample
    if len(data) < 5:
        return None, None, None
    
    X = data.select(feature_cols).to_numpy()
    X_scaled = StandardScaler().fit_transform(X)
    
    
    # fit PCA with 3 components for 3D visualiation
    pca = PCA(n_components=3)  
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    
    return X_pca, var_exp, np.arange(len(data))
    
    
    
'''
Function used to loop over all passes and tools and call the 'compute_pca_3d' function to plot the corresponding pca
'''
def plot(filename=""):
    os.makedirs("images", exist_ok=True)
    
    
    passes = ["Pre-Finishing","Finishing"]
    result = pl.scan_csv(filename)
    result = result.filter(pl.col("pass_type").is_in(passes))
    tools = result.select(pl.col("ToolIdx")).unique().collect()["ToolIdx"].to_list()
    tools.sort()
    pass_types = result.select(pl.col("pass_type")).unique().collect()["pass_type"].to_list() # get all unique pass dynamically (should be the same as 'passes')
    for p in pass_types:
        for t in tools:
            data = result.filter((pl.col("ToolIdx") == t) & (pl.col("pass_type") == p))
            X_pca, var_exp, indices = compute_pca_3d(data)
            timestamp = data.select("timestamp").collect()["timestamp"].to_list()
            
            if X_pca is not None:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # 3D scatter colored by time to show progression along the pass
                sc = ax.scatter(
                    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                    c=timestamp, 
                    cmap="viridis", 
                    s=25, 
                    alpha=0.8,
                    edgecolors='w',
                    linewidths=0.2
                )
                ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], color='black', alpha=0.15, linewidth=1)
                
                var_totale = np.sum(var_exp) * 100
                ax.set_title(f"PCA 3D - Outil {t}\pass type : {p} (Variance expliquée : {var_totale:.1f}%)", fontsize=12, fontweight='bold')
                ax.set_xlabel(f"PC1 ({float(var_exp[0]*100)}%)")
                ax.set_ylabel(f"PC2 ({float(var_exp[1]*100)}%)")
                ax.set_zlabel(f"PC3 ({float(var_exp[2]*100)}%)")
                
                cbar = fig.colorbar(sc, ax=ax, pad=0.1)
                cbar.set_label("Progression temporelle (timestamp)")
                
                plt.tight_layout()
                
                fig_name = f"images/PCA_Outil_{t}_{p}.png"
                #plt.savefig(fig_name, dpi=150)
                plt.show()
                plt.close(fig)

def main():
    plot( "../data/v5/tsfel_extracted_v5.csv")

if __name__ == "__main__":
    main()