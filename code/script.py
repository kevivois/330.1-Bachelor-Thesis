import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq

initial_path = "/run/media/kevivois/T7/BACHELOR"

FS = 20000        
NPERSEG = 1024   
VMIN, VMAX = -80, 0

filename1 = initial_path + "/out/pass_000013.parquet"
filename2 = initial_path + "/out/pass_000086.parquet"

def plot_comparison(file_new, file_worn):
    df_new = pd.read_parquet(file_new)
    df_worn = pd.read_parquet(file_worn)
    
    sound_new = df_new["Sound"].to_numpy()
    sound_worn = df_worn["Sound"].to_numpy()
    
    accelerator_x_new = df_new["AccX"].to_numpy()
    accelerator_y_new = df_new["AccY"].to_numpy()
    accelerator_z_new = df_new["AccZ"].to_numpy()
    
    accelerometre_new = np.sqrt(accelerator_x_new**2 + accelerator_y_new**2 + accelerator_z_new**2)
    accelerometre_new -= np.mean(accelerometre_new)
    
    accelerator_x_worn = df_worn["AccX"].to_numpy()
    accelerator_y_worn = df_worn["AccY"].to_numpy()
    accelerator_z_worn = df_worn["AccZ"].to_numpy()
    
    accelerometre_worn = np.sqrt(accelerator_x_worn**2 + accelerator_y_worn**2 + accelerator_z_worn**2)
    accelerometre_worn -= np.mean(accelerometre_worn)
    
    s_new_c = sound_new - np.mean(sound_new)
    s_worn_c = sound_worn - np.mean(sound_worn)
    
    n_n = len(s_new_c)
    n_w = len(s_worn_c)
    
    freq_n = fftfreq(n_n, 1/FS)
    fft_n = np.abs(fft(s_new_c))
    
    freq_w = fftfreq(n_w, 1/FS)
    fft_w = np.abs(fft(s_worn_c))
    
    fft_n_db = 20 * np.log10(fft_n + 1e-10)
    fft_w_db = 20 * np.log10(fft_w + 1e-10)
    
    fft_n_a = np.abs(fft(accelerometre_new))
    fft_n_a_db = 20 * np.log10(fft_n_a + 1e-10)
    
    fft_w_a = np.abs(fft(accelerometre_worn))
    fft_w_a_db = 20 * np.log10(fft_w_a + 1e-10)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    axes[0, 0].plot(np.arange(n_n)/FS, s_new_c)
    axes[0, 0].set_title("Signal Temporel - Outil Neuf")
    
    axes[0, 1].plot(freq_n[:n_n//2], fft_n_db[:n_n//2])
    axes[0, 1].set_title("FFT - Outil Neuf")
    
    axes[1, 0].plot(np.arange(n_w)/FS, s_worn_c, color='red')
    axes[1, 0].set_title("Signal Temporel - Outil Usé")
    
    axes[1, 1].plot(freq_w[:n_w//2], fft_w_db[:n_w//2], color='red')
    axes[1, 1].set_title("FFT - Outil Usé")
    
    axes[0, 2].plot(freq_n[:n_n//2], fft_n_a_db[:n_n//2], color="green")
    axes[0, 2].set_title("Accéléromètre FFT - Outil Neuf")
    
    axes[1, 2].plot(freq_w[:n_w//2], fft_w_a_db[:n_w//2], color="red")
    axes[1, 2].set_title("Accéléromètre FFT - Outil Usé")
    
    plt.tight_layout()
    plt.show()

plot_comparison(filename1, filename2)