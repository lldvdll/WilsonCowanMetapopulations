"""
Wilson-Cowan Metapopulation Experiment Pipeline
Continuous Loop: Random Search -> True Binary Boundary Search -> Topology Evaluation
"""

import numpy as np
import networkx as nx
import pandas as pd
from scipy.signal import find_peaks, hilbert
import os
import random
import sys

# --- GLOBALS & SETTINGS ---
CSV_FILE = "experiment_results.csv"
DURATION = 400.0
DT = 0.1
N_NODES = 16
TRANSIENT_CUTOFF = 20.0  
NOISE_LEVEL = 0.02
BINARY_SEARCH_STEPS = 5   
MAX_ROWS = 100000          # Failsafe to prevent infinite file size

PARAM_BOUNDS = {
    'c_ee': (-25.0, 25.0), 'c_ei': (-25.0, 25.0), 
    'c_ie': (-25.0, 25.0), 'c_ii': (-25.0, 25.0),
    'P': (-10.0, 10.0), 'Q': (-10.0, 10.0),
    'tau_1': (0.1, 50.0), 'tau_2': (0.1, 50.0), 'rho': (0.1, 50.0),
    'beta': (1.0, 20.0), 'k': (-2.0, 20.0), 'alpha': (0.01, 5.0)
}

# Global counter for CSV rows
csv_row_count = 0

# --- 1. NETWORK & SIMULATOR ---
def get_normalized_matrix(N, topology):
    if topology == 'line': G = nx.path_graph(N)
    elif topology == 'lattice':
        side = int(np.ceil(np.sqrt(N)))
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
        G.remove_nodes_from(list(G.nodes())[N:])
    elif topology == 'full': G = nx.complete_graph(N)
    else: # STRICT DEGREE 2 RING
        G = nx.circulant_graph(N, [1])
        
    W = nx.to_numpy_array(G)
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return W / row_sums[:, np.newaxis]

def run_simulation(p, topology='ring'):
    W = get_normalized_matrix(N_NODES, topology)
    steps = int(DURATION / DT)
    
    E = np.full((steps, N_NODES), 0.25)
    I = np.full((steps, N_NODES), 0.75)
    
    t1 = max(1, int(p['tau_1'] / DT))
    t2 = max(1, int(p['tau_2'] / DT))
    tr = max(1, int(p['rho'] / DT))
    max_delay = max(t1, t2, tr)
    
    np.random.seed(42) 
    E[:max_delay, :] += np.random.uniform(-NOISE_LEVEL, NOISE_LEVEL, N_NODES)
    I[:max_delay, :] += np.random.uniform(-NOISE_LEVEL, NOISE_LEVEL, N_NODES)

    for i in range(max_delay, steps - 1):
        coupling = W @ E[i - tr] 
        e_arg = p['c_ee']*E[i-t1] + p['c_ie']*I[i-t2] + p['P'] + p['k']*coupling
        i_arg = p['c_ei']*E[i-t2] + p['c_ii']*I[i-t1] + p['Q']
        
        dE = -E[i] + 1.0 / (1.0 + np.exp(-p['beta'] * np.clip(e_arg, -10, 10)))
        dI = p['alpha'] * (-I[i] + 1.0 / (1.0 + np.exp(-p['beta'] * np.clip(i_arg, -10, 10))))
        
        E[i+1] = E[i] + dE * DT
        I[i+1] = I[i] + dI * DT
        
    return E, I

# --- 2. STRICT METRICS ENGINE ---
def calculate_metrics(E, I):
    cutoff_idx = int(TRANSIENT_CUTOFF / DT)
    if cutoff_idx >= len(E): cutoff_idx = 0 
    
    E_steady = E[cutoff_idx:, :]
    I_steady = I[cutoff_idx:, :]
    eval_time = (DURATION - TRANSIENT_CUTOFF)
    
    # 1. Peak Detection with Prominence
    e_peaks, _ = find_peaks(E_steady[:, 0], distance=10, prominence=0.05)
    i_peaks, _ = find_peaks(I_steady[:, 0], distance=10, prominence=0.05)
    n_E_peaks, n_I_peaks = len(e_peaks), len(i_peaks)
    
    # 2. Sustained Amplitude Check
    end_window = int(50.0 / DT)
    E_end_amp = np.max(E_steady[-end_window:, 0]) - np.min(E_steady[-end_window:, 0])
    I_end_amp = np.max(I_steady[-end_window:, 0]) - np.min(I_steady[-end_window:, 0])
    is_sustained = (E_end_amp > 0.05) and (I_end_amp > 0.05)
    
    # 3. Strict Oscillation Flag
    osc_detected = (n_E_peaks >= 3) and (n_I_peaks >= 3) and is_sustained
    
    E_max, E_min = float(np.max(E_steady[:, 0])), float(np.min(E_steady[:, 0]))
    I_max, I_min = float(np.max(I_steady[:, 0])), float(np.min(I_steady[:, 0]))
    E_amp, I_amp = E_max - E_min, I_max - I_min
    
    freq_E = n_E_peaks / (eval_time / 1000.0) if n_E_peaks > 1 else 0
    freq_I = n_I_peaks / (eval_time / 1000.0) if n_I_peaks > 1 else 0
    peak_ratio = (n_I_peaks / n_E_peaks) if n_E_peaks > 0 else 0
    
    plv_E = 0
    if osc_detected:
        E_centered = E_steady - np.mean(E_steady, axis=0)
        E_centered += np.random.normal(0, 1e-6, E_centered.shape) 
        phases = np.angle(hilbert(E_centered, axis=0))
        mean_phase_vector = np.mean(np.exp(1j * phases), axis=1)
        plv_E = float(np.mean(np.abs(mean_phase_vector)))

    return {
        'Oscillation_Detected': bool(osc_detected),
        'E_max': E_max, 'E_min': E_min, 'E_amp': E_amp,
        'I_max': I_max, 'I_min': I_min, 'I_amp': I_amp,
        'E_peaks': n_E_peaks, 'I_peaks': n_I_peaks,
        'Peak_Ratio_I_E': peak_ratio,
        'Freq_E': freq_E, 'Freq_I': freq_I,
        'PLV_E': plv_E
    }

# --- 3. CSV LOGGING ---
def log_result(phase, topology, params, metrics):
    global csv_row_count
    row = {'Phase': phase, 'Topology': topology, 'Duration': DURATION, 'N_Nodes': N_NODES}
    row.update(params)
    row.update(metrics)
    
    df = pd.DataFrame([row])
    write_header = not os.path.exists(CSV_FILE)
    df.to_csv(CSV_FILE, mode='a', header=write_header, index=False)
    csv_row_count += 1
    
    if csv_row_count >= MAX_ROWS:
        print(f"\n[!] MAX_ROWS ({MAX_ROWS}) reached. Stopping experiment safely.")
        sys.exit(0)

# --- 4. THE PIPELINE ---
def generate_random_params():
    return {k: random.uniform(v[0], v[1]) for k, v in PARAM_BOUNDS.items()}

def phase_1_random_search(iteration):
    p = generate_random_params()
    E, I = run_simulation(p, topology='ring')
    metrics = calculate_metrics(E, I)
    
    log_result('1_Random_Search', 'ring', p, metrics)
    
    if metrics['Oscillation_Detected']:
        print(f"  [!] Oscillation Found at Iteration {iteration}! Proceeding to Phase 2...")
        return p
    return None

def phase_2_binary_boundary_search(center_params):
    print("  --- PHASE 2: Divide and Conquer Boundary Search ---")
    boundaries = []
    
    for param, (low_bound, high_bound) in PARAM_BOUNDS.items():
        # --- Search UP ---
        test_p = center_params.copy()
        low = center_params[param]
        high = high_bound
        
        for step in range(BINARY_SEARCH_STEPS):
            mid = low + (high - low) / 2.0
            test_p[param] = mid
            
            E, I = run_simulation(test_p, topology='ring')
            metrics = calculate_metrics(E, I)
            log_result(f'2_Boundary_Search_{param}_UP', 'ring', test_p, metrics)
            
            if metrics['Oscillation_Detected']: low = mid 
            else: high = mid  
                
        upper_boundary_p = center_params.copy()
        upper_boundary_p[param] = low
        boundaries.append(upper_boundary_p)
        
        # --- Search DOWN ---
        test_p = center_params.copy()
        low = low_bound
        high = center_params[param]
        
        for step in range(BINARY_SEARCH_STEPS):
            mid = low + (high - low) / 2.0
            test_p[param] = mid
            
            E, I = run_simulation(test_p, topology='ring')
            metrics = calculate_metrics(E, I)
            log_result(f'2_Boundary_Search_{param}_DOWN', 'ring', test_p, metrics)
            
            if metrics['Oscillation_Detected']: high = mid  
            else: low = mid   
                
        lower_boundary_p = center_params.copy()
        lower_boundary_p[param] = high
        boundaries.append(lower_boundary_p)
        
    return boundaries

def phase_3_topology_test(param_sets):
    print("  --- PHASE 3: Topology Evaluation on Boundaries ---")
    topologies = ['ring', 'line', 'lattice', 'full']
    
    for p in param_sets:
        for top in topologies:
            E, I = run_simulation(p, topology=top)
            metrics = calculate_metrics(E, I)
            log_result('3_Topology_Test', top, p, metrics)


# --- EXECUTE SCRIPT ---
if __name__ == "__main__":
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE) # Clear old runs
        
    print("Starting Continuous Experiment...")
    print(f"Press Ctrl+C to stop safely. Max rows set to {MAX_ROWS}.")
    
    iteration = 0
    try:
        while True:
            iteration += 1
            if iteration % 100 == 0:
                print(f"Random Search Iteration {iteration}...")
                
            # Step 1: Run a single random search
            center = phase_1_random_search(iteration)
            
            # Step 2 & 3: If it oscillates, map the boundaries and test topologies
            if center is not None:
                bounds = phase_2_binary_boundary_search(center)
                phase_3_topology_test([center] + bounds)
                print(f"Completed mapping for Oscillation {iteration}. Resuming Random Search...")
                
    except KeyboardInterrupt:
        print(f"\n\n[!] Experiment manually stopped by user (Ctrl+C).")
    finally:
        print(f"Total Rows Logged: {csv_row_count}")
        print(f"Results successfully saved to {CSV_FILE}")