import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq

initial_path = "/run/media/kevivois/T7/BACHELOR"
FS = 20000        
filename1 = initial_path + "/out/pass_000013.parquet"
filename2 = initial_path + "/out/pass_000086.parquet"

def plot_comparison(file_new, file_worn):
    df_new = pd.read_parquet(file_new)
    df_worn = pd.read_parquet(file_worn)
    
    s_n = (df_new["Sound"].to_numpy() - np.mean(df_new["Sound"]))
    s_w = (df_worn["Sound"].to_numpy() - np.mean(df_worn["Sound"]))
    
    # Norme accéléromètre
    a_n = np.sqrt(df_new["AccX"]**2 + df_new["AccY"]**2 + df_new["AccZ"]**2)
    a_n = (a_n - np.mean(a_n))
    
    a_w = np.sqrt(df_worn["AccX"]**2 + df_worn["AccY"]**2 + df_worn["AccZ"]**2)
    a_w = (a_w - np.mean(a_w))
    
    # Calcul FFT et Normalisation par le max
    def get_norm_fft_db(signal):
        n = len(signal)
        fft_vals = np.abs(fft(signal))[:n//2]
        fft_db = 20 * np.log10(fft_vals / (np.max(fft_vals) + 1e-10))
        return fftfreq(n, 1/FS)[:n//2], fft_db

    f_sn, fft_sn = get_norm_fft_db(s_n)
    f_sw, fft_sw = get_norm_fft_db(s_w)
    f_an, fft_an = get_norm_fft_db(a_n)
    f_aw, fft_aw = get_norm_fft_db(a_w)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].plot(f_sn, fft_sn, label="Sound (Neuf)", alpha=0.7)
    axes[0].plot(f_an, fft_an, label="Accéléromètre (Neuf)", alpha=0.7)
    axes[0].set_title("Comparaison Neuf : Sound vs Accéléromètre")
    axes[0].legend()
    axes[0].set_ylabel("Amplitude Normalisée (dB)")
    
    axes[1].plot(f_sw, fft_sw, label="Sound (Usé)", color='red', alpha=0.7)
    axes[1].plot(f_aw, fft_aw, label="Accéléromètre (Usé)", color='green', alpha=0.7)
    axes[1].set_title("Comparaison Usé : Sound vs Accéléromètre")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

plot_comparison(filename1, filename2)