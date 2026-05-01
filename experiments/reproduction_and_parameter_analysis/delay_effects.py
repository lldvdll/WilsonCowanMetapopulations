import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.metapopulation import Metapopulation

# =========================================================
# CONFIGURATION CONSTANTS
# =========================================================
FREQ_MULTIPLIER = 1000.0 
MAX_PLOT_FREQ = 150.0   

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def compute_network_fft(trajectories, dt):
    """
    Computes the FFT. Uses std filter for flatlines and 
    np.argmax for robust peak detection (eliminates artifacts).
    """
    E_nodes = trajectories[0]
    I_nodes = trajectories[1]
    
    N_nodes, steps = E_nodes.shape
    freqs = np.fft.rfftfreq(steps, d=dt) * FREQ_MULTIPLIER
    window = np.hanning(steps)
    
    E_spectra = []
    I_spectra = []
    
    for n in range(N_nodes):
        # 1. Flatline Check
        if np.std(E_nodes[n]) < 0.01:
            E_spectra.append(np.zeros_like(freqs))
        else:
            E_sig = (E_nodes[n] - np.mean(E_nodes[n])) * window
            E_spectra.append(np.abs(np.fft.rfft(E_sig)))
            
        if np.std(I_nodes[n]) < 0.01:
            I_spectra.append(np.zeros_like(freqs))
        else:
            I_sig = (I_nodes[n] - np.mean(I_nodes[n])) * window
            I_spectra.append(np.abs(np.fft.rfft(I_sig)))
        
    E_spectra = np.array(E_spectra)
    I_spectra = np.array(I_spectra)
    mean_E_spectrum = np.mean(E_spectra, axis=0)
    mean_I_spectrum = np.mean(I_spectra, axis=0)
    
    # 2. Robust Peak Detection (No find_peaks brittleness)
    def get_dominant_freq(spectrum):
        if np.max(spectrum) < 1e-6: 
            return 0.0 
        # Simply return the frequency bin with the most energy
        return freqs[np.argmax(spectrum)]
        
    dom_freq_E = get_dominant_freq(mean_E_spectrum)
    dom_freq_I = get_dominant_freq(mean_I_spectrum)
        
    return freqs, E_spectra, I_spectra, mean_E_spectrum, mean_I_spectrum, dom_freq_E, dom_freq_I

def load_config_from_csv(path, row_index):
    df = pd.read_csv(path)
    return df.iloc[row_index - 2].to_dict()

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    script_path = os.path.dirname(os.path.realpath(__file__))
    csv_config_path = os.path.join(script_path, 'parameter_reference.csv')
    base_config_path = os.path.join(script_path, 'network_effects.yaml')
    
    base_params = load_config_from_csv(csv_config_path, 19)
    base_params['topology'] = 'ring'
    base_params['N'] = int(base_params['N'])
    
    duration = 1000.0
    dt = 0.1
    w_size = 250.0  
    
    # ---------------------------------------------------------
    # PART 1: SINGLE LONG SIMULATION 
    # ---------------------------------------------------------
    print("Running base simulation for trajectory and FFT windows...")
    model = Metapopulation()
    model.load_config(base_config_path)
    model.create_network(base_params)
    
    sim_params = base_params.copy()
    sim_params['duration'] = duration
    model.initialise_model(sim_params)
    model.config['simulation']['duration'] = duration
    model.run_simulation()
    
    t = model.model.time_array
    traj = model.model.trajectories
    
    w1_end_idx = int(w_size / dt)
    w2_start_idx = int((duration - w_size) / dt)
    
    traj_w1 = traj[:, :, :w1_end_idx]
    traj_w2 = traj[:, :, w2_start_idx:]
    
    f1, E_sp1, I_sp1, mE1, mI1, domE1, domI1 = compute_network_fft(traj_w1, dt)
    f2, E_sp2, I_sp2, mE2, mI2, domE2, domI2 = compute_network_fft(traj_w2, dt)
    
    # ---------------------------------------------------------
    # PART 2: HEATMAP SWEEPS (LINEAR SPACING)
    # ---------------------------------------------------------
    grid_size = 20  # Linear grid, 20x20 is usually a good balance
    tau_values = np.linspace(0.1, 20.0, grid_size)
    rho_values = np.linspace(0.1, 80.0, grid_size) 
    
    hm_E_w1, hm_E_w2 = np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size))
    hm_I_w1, hm_I_w2 = np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size))
    
    print(f"Running parameter sweep ({grid_size}x{grid_size} grid)...")
    for i, rho_val in enumerate(rho_values):
        for j, tau_val in enumerate(tau_values):
            sweep_params = base_params.copy()
            sweep_params['tau_1'] = tau_val
            sweep_params['tau_2'] = tau_val
            sweep_params['rho'] = rho_val
            
            model.initialise_model(sweep_params)
            model.run_simulation()
            
            swp_traj = model.model.trajectories
            swp_w1 = swp_traj[:, :, :w1_end_idx]
            swp_w2 = swp_traj[:, :, w2_start_idx:]
            
            _, _, _, _, _, dE1, dI1 = compute_network_fft(swp_w1, dt)
            _, _, _, _, _, dE2, dI2 = compute_network_fft(swp_w2, dt)
            
            hm_E_w1[i, j], hm_I_w1[i, j] = dE1, dI1
            hm_E_w2[i, j], hm_I_w2[i, j] = dE2, dI2

    # ---------------------------------------------------------
    # PART 3: PLOTTING LAYOUT
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(14, 18))
    # fig.suptitle("Network Frequency Analysis & Parameter Space Heatmaps", fontsize=18, y=0.98)
    
    gs = GridSpec(4, 2, height_ratios=[1.5, 1.2, 1.5, 1.5], hspace=0.35)
    ax_traj = fig.add_subplot(gs[0, :])
    ax_fft1 = fig.add_subplot(gs[1, 0])
    ax_fft2 = fig.add_subplot(gs[1, 1])
    ax_hm_E1 = fig.add_subplot(gs[2, 0])
    ax_hm_E2 = fig.add_subplot(gs[2, 1])
    ax_hm_I1 = fig.add_subplot(gs[3, 0])
    ax_hm_I2 = fig.add_subplot(gs[3, 1])

    # --- Plot 1: Trajectory ---
    for n in range(base_params['N']):
        ax_traj.plot(t, traj[0, n], color='blue', alpha=0.2, linewidth=1)
        ax_traj.plot(t, traj[1, n], color='black', alpha=0.2, linewidth=1)
    ax_traj.axvspan(0, w_size, color='red', alpha=0.1, label='Window 1 (Start)')
    ax_traj.axvspan(duration - w_size, duration, color='green', alpha=0.1, label='Window 2 (End)')
    ax_traj.set_title("Network Trajectories (E=Blue, I=Black)", fontsize=14)
    ax_traj.set_xlim(0, duration)
    ax_traj.set_ylim(-0.05, 1.05)
    ax_traj.legend(loc="upper right")

    # --- Plot 2 & 3: FFT Spectra ---
    fft_data = [
        (ax_fft1, f1, E_sp1, I_sp1, mE1, mI1, domE1, domI1, "Window 1 FFT (Transient)"),
        (ax_fft2, f2, E_sp2, I_sp2, mE2, mI2, domE2, domI2, "Window 2 FFT (Steady State)")
    ]
    for ax, freqs, E_sp, I_sp, mE, mI, dE, dI, title in fft_data:
        ax.plot(freqs, mE, color='blue', linewidth=2, label=f'Mean E (Peak: {dE:.1f} Hz)')
        ax.plot(freqs, mI, color='black', linewidth=2, label=f'Mean I (Peak: {dI:.1f} Hz)')
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, MAX_PLOT_FREQ) 
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.legend()

    # --- Plot 4 to 7: Heatmaps (Linear & Separated Scales) ---
    # Find sensible maximums to prevent outliers from washing out the color scale
    # We cap E at 100 Hz and I at 200 Hz (or their actual maximums if lower)
    vmax_E = min(np.max([hm_E_w1, hm_E_w2]), 100.0)
    vmax_I = min(np.max([hm_I_w1, hm_I_w2]), 200.0)

    # Create a colormap where 0.0 (flatline) is explicitly black for contrast
    cmap = plt.cm.viridis.copy()
    cmap.set_under(color='black')

    # Excitatory
    for ax, hm, title in [(ax_hm_E1, hm_E_w1, "Excitatory (E) - Window 1"), 
                          (ax_hm_E2, hm_E_w2, "Excitatory (E) - Window 2")]:
        im = ax.imshow(hm, origin='lower', aspect='auto', cmap=cmap,
                       extent=[tau_values[0], tau_values[-1], rho_values[0], rho_values[-1]],
                       vmin=0.1, vmax=vmax_E) # vmin=0.1 forces 0.0 to trigger the 'under' color (black)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"Intra-node delay ($\tau$)")
        ax.set_ylabel(r"Inter-node delay ($\rho$)")
        fig.colorbar(im, ax=ax, label="Frequency (Hz)", extend='max')

    # Inhibitory
    for ax, hm, title in [(ax_hm_I1, hm_I_w1, "Inhibitory (I) - Window 1"), 
                          (ax_hm_I2, hm_I_w2, "Inhibitory (I) - Window 2")]:
        im = ax.imshow(hm, origin='lower', aspect='auto', cmap=cmap,
                       extent=[tau_values[0], tau_values[-1], rho_values[0], rho_values[-1]],
                       vmin=0.1, vmax=vmax_I)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"Intra-node delay ($\tau$)")
        ax.set_ylabel(r"Inter-node delay ($\rho$)")
        fig.colorbar(im, ax=ax, label="Frequency (Hz)", extend='max')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(script_path, "figure_7_reproduction.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()