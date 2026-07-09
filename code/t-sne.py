# This script reads the extracted CSV dataset, filters it by tool and pass type, and computes a 2D t-SNE embedding to visualize how the feature space evolves over time.


import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# Compute a 2D t-SNE embedding for one subset of the data.
def compute_tsne_2d(data:pl.LazyFrame):
    data_pl = data.collect()
    if len(data_pl) == 0:
        return None, None
    
    # Columns that are not used as features
    meta_cols = [
        "id", "sensor_file", "timestamp", "time", "ToolIdx", "PassNumber", 
        "pass_type", "y", "DB_PASSES/NUMERO_OF", 
        "DB_PASSES/SELECTION_ALLIAGE", "PassID", "start_pos", "end_pos", 
        "DB_PASSES/NUMERO_PASSE", "timestamp_right", "index","DB_PASSES/COMPTEUR_RESET_PASSES"
    ]
    feature_cols = [c for c in data_pl.columns if c not in meta_cols]
    
    # Skip too-small subsets
    if len(data_pl) < 40:
        return None, None
    
    # Standardize features before t-SNE
    X = data_pl.select(feature_cols).to_numpy()
    X_scaled = StandardScaler().fit_transform(X)
    
    
     # Perplexity must stay smaller than the number of sampless
    perp = min(30, len(data_pl) - 10)
    
    
    # Fit t-SNE in 2D
    tsne = TSNE(
        n_components=2, 
        perplexity=perp, 
        random_state=42, 
        n_jobs=-1,        
        max_iter=1000,
        init='pca'        
    )
    
    X_tsne = tsne.fit_transform(X_scaled)
    
    
     # Return embedding and simple indices for coloring
    return X_tsne, np.arange(len(data_pl))
    
    
# Loop over tools and pass types, compute t-SNE, and plot the results.
def plot_tsne(filename=""):
    os.makedirs("images_tsne", exist_ok=True)
    
    result = pl.scan_csv(filename)
    
    tools = result.select(pl.col("ToolIdx").unique()).collect().get_column("ToolIdx").to_list()
    tools.sort()
    passes = ["Finishing","Pre-Finishing"]
    result = result.filter(pl.col("pass_type").is_in(passes))
    pass_types = result.select(pl.col("pass_type").unique()).collect().get_column("pass_type").to_list()
    
    for p in pass_types:
        for t in tools:

            if p is None:
                continue
                
            print(f"Calcul t-SNE en cours pour l'Outil {t} - Régime {p}...")
            data = result.filter((pl.col("ToolIdx") == t) & (pl.col("pass_type") == p)).sort("timestamp")
            X_tsne, indices = compute_tsne_2d(data)
            
            if X_tsne is not None:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                sc = ax.scatter(
                    X_tsne[:, 0], X_tsne[:, 1], 
                    c=indices, 
                    cmap="turbo", 
                    s=30, 
                    alpha=0.8,
                    edgecolors='w',
                    linewidths=0.3
                )
                
                ax.plot(X_tsne[:, 0], X_tsne[:, 1], color='black', alpha=0.15, linewidth=0.8)
                
                ax.set_title(f"Collectif Manifold t-SNE 2D - Outil {t}\nRégime : {p}", fontsize=12, fontweight='bold')
                ax.set_xlabel("t-SNE Dimension 1")
                ax.set_ylabel("t-SNE Dimension 2")
                
                ax.grid(True, linestyle="--", alpha=0.5)
                
                cbar = fig.colorbar(sc, ax=ax, pad=0.05)
                cbar.set_label("Progression (Index des fenêtres temporelles)")
                
                plt.tight_layout()
                
                safe_p_name = p.replace(" ", "_").replace("/", "_")
                fig_name = f"tSNE_Outil_{t}_{safe_p_name}.png"
                
                #plt.savefig(f"images/v2/pca/{fig_name}")
                
                plt.show()
                plt.close(fig)

def main():
    plot_tsne("../data/v4/tsfel_extracted_new.csv")

if __name__ == "__main__":
    main()