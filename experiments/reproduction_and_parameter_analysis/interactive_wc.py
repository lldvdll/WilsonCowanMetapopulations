"""
Streamlined Interactive Wilson-Cowan Simulator
- Uses the Metapopulation framework directly for 100% exact algorithm matching.
- Metrics disabled by default for speed.
- Sliders only (no textboxes).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from scipy.signal import find_peaks, hilbert
from src.metapopulation import Metapopulation

# ==========================================
# 1. STARTING PARAMETERS
# ==========================================
# Edit these vertically listed defaults to change the starting state
STARTING_PARAMS = {  # Fig2c corrected
    'c_ee': -6.,
    'c_ei': -3.5,
    'c_ie': 2.5,
    'c_ii': -6.,
    'P': 0.2,
    'Q': 0.2,
    'tau_1': 1.0,
    'tau_2': 1.4,
    'rho': 10.0,
    'beta': 10.0,
    'k': 11.,
    'alpha': 0.6,
    'N': 2,
    'duration': 100.0,
    'topology': 'full',
    'jitter': 0.1
}
STARTING_PARAMS = {  # Fig3 corrected
    'c_ee': 3.7,
    'c_ei': -7.3,
    'c_ie': 18.,
    'c_ii': -11.,
    'P': -6.7,
    'Q': 6.5,
    'tau_1': 1,
    'tau_2': 4.7,
    'rho': 10.0,
    'beta': 10.0,
    'k': 0.5,
    'alpha': 0.6,
    'N': 16,
    'duration': 140.0,
    'topology': 'full',
    'jitter': 0.1
}
STARTING_PARAMS = {  # Fig3 corrected
    'c_ee': 3.78,
    'c_ei': -6.51,
    'c_ie': 19.4,
    'c_ii': -9.38,
    'P': -7.97,
    'Q': 4.8,
    'tau_1': 0.78,
    'tau_2': 5.08,
    'rho': 10.0,
    'beta': 10.0,
    'k': 0.5,
    'alpha': 0.6,
    'N': 16,
    'duration': 140.0,
    'topology': 'ring',
    'jitter': 0.1
}
STARTING_PARAMS = {  # Fig3 corrected
    'c_ee': 3.78,
    'c_ei': -6.51,
    'c_ie': 19.4,
    'c_ii': -9.38,
    'P': -7.97,
    'Q': 4.8,
    'tau_1': 0.78,
    'tau_2': 5.08,
    'rho': 10.0,
    'beta': 10.0,
    'k': 0.5,
    'alpha': 0.6,
    'N': 16,
    'duration': 140.0,
    'topology': 'ring',
    'jitter': 0.0
}


def load_config_from_csv(path, row):
    df = pd.read_csv(path)
    return df.iloc[row-2].to_dict()
row = 19
csv_config_path = os.path.join(os.path.dirname(__file__), 'parameter_reference.csv')
STARTING_PARAMS = load_config_from_csv(csv_config_path, row)

dt = 0.1 # Fixed fine timestep for explicit Euler accuracy
TRANSIENT_CUTOFF = 0.0 

# ==========================================
# 2. INITIALIZE METAPOPULATION FRAMEWORK 
# ==========================================
# We use your actual framework to guarantee the exact same algorithm is executed
model_framework = Metapopulation()
model_framework.config = {
    "model_params": {
        "mode": "wilson_cowan_efficient",
        "c_ee": STARTING_PARAMS['c_ee'],
        "c_ei": STARTING_PARAMS['c_ei'],
        "c_ie": STARTING_PARAMS['c_ie'],
        "c_ii": STARTING_PARAMS['c_ii'],
        "P": STARTING_PARAMS['P'],
        "Q": STARTING_PARAMS['Q'],
        "tau_1": STARTING_PARAMS['tau_1'],
        "tau_2": STARTING_PARAMS['tau_2'],
        "rho": STARTING_PARAMS['rho'],
        "beta": STARTING_PARAMS['beta'],
        "k": STARTING_PARAMS['k'],
        "alpha": STARTING_PARAMS['alpha'],
        "T_e": 1.0,
        "T_i": 1.0
    },
    "network_params": {
        "topology": STARTING_PARAMS['topology'],
        "N": STARTING_PARAMS['N'],
        "normalise": False,
        "p": None
    },
    "simulation": {
        "duration": STARTING_PARAMS['duration'],
        "dt": dt,
        "initial_conditions": {"E": 0.25, "I": 0.75},
        "jitter": STARTING_PARAMS['jitter']
    }
}
# (Simulation runs for the first time inside the update() loop)


# ==========================================
# 3. PLOTTING SETUP
# ==========================================
fig, ax = plt.subplots(figsize=(16, 10))
plt.subplots_adjust(left=0.3, bottom=0.5, right=0.6) 
ax.set_title("Vectorized Wilson-Cowan DDE Simulator (Metapopulation Native)")
ax.set_xlabel("Time")
ax.set_ylabel("Activity")
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(0, STARTING_PARAMS['duration'])
ax.grid(True, alpha=0.3)

lines_E = []
lines_I = []
current_N = 0
current_duration = 0

# Diagnostics Text Panel
ax_metrics = plt.axes([0.76, 0.45, 0.23, 0.45])
ax_metrics.axis('off')
txt_metrics = ax_metrics.text(0, 1, "Metrics Disabled...", fontsize=10, va='top', family='monospace')

# Metrics Toggle (Defaults to False)
ax_toggle = plt.axes([0.76, 0.9, 0.15, 0.05])
check_metrics = CheckButtons(ax_toggle, ['Calculate Metrics (Slow)'], [False])


# ==========================================
# 4. METRICS ENGINE
# ==========================================
def calculate_metrics(E, I, duration):
    cutoff_idx = int(TRANSIENT_CUTOFF / dt)
    if cutoff_idx >= len(E): cutoff_idx = 0 
    
    E_steady = E[cutoff_idx:, :]
    I_steady = I[cutoff_idx:, :]
    
    e_peaks, _ = find_peaks(E_steady[:, 0], distance=10, prominence=0.05)
    i_peaks, _ = find_peaks(I_steady[:, 0], distance=10, prominence=0.05)
    n_E_peaks, n_I_peaks = len(e_peaks), len(i_peaks)
    
    end_window = int(50.0 / dt)
    if len(E_steady) > end_window:
        E_end_amp = np.max(E_steady[-end_window:, 0]) - np.min(E_steady[-end_window:, 0])
        I_end_amp = np.max(I_steady[-end_window:, 0]) - np.min(I_steady[-end_window:, 0])
    else:
        E_end_amp, I_end_amp = 0, 0
        
    is_sustained = (E_end_amp > 0.05) and (I_end_amp > 0.05)
    osc_detected = (n_E_peaks >= 3) and (n_I_peaks >= 3) and is_sustained
    
    E_amp = float(np.max(E_steady[:, 0])) - float(np.min(E_steady[:, 0]))
    I_amp = float(np.max(I_steady[:, 0])) - float(np.min(I_steady[:, 0]))
    
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
        'E_amp': E_amp, 'I_amp': I_amp,
        'E_peaks': n_E_peaks, 'I_peaks': n_I_peaks,
        'Peak_Ratio_I_E': peak_ratio,
        'PLV_E': plv_E
    }


# ==========================================
# 5. FAST SLIDERS (No TextBoxes)
# ==========================================
sliders = {}
def make_control(name, y_pos, min_val, max_val, init_val, is_right=False, valstep=None):
    x_base = 0.50 if is_right else 0.15
    ax_slider = plt.axes([x_base, y_pos, 0.20, 0.02])
    sl = Slider(ax_slider, name, min_val, max_val, valinit=init_val, valstep=valstep)
    sl.on_changed(lambda val: update(None)) 
    sliders[name] = sl
    return sl

valstep1 = 0.01
RANGE = 'wide'  # wide or narrow

# Wide range
if RANGE == 'wide':
    make_control('c_ee',  0.40, -25, 25, STARTING_PARAMS['c_ee'], valstep=valstep1)
    make_control('c_ei',  0.35, -25, 25, STARTING_PARAMS['c_ei'], valstep=valstep1)
    make_control('c_ie',  0.30, -25, 25, STARTING_PARAMS['c_ie'], valstep=valstep1)
    make_control('c_ii',  0.25, -25, 25, STARTING_PARAMS['c_ii'], valstep=valstep1)
    make_control('P',     0.20, -10, 10, STARTING_PARAMS['P'], valstep=valstep1)
    make_control('Q',     0.15, -10, 10, STARTING_PARAMS['Q'], valstep=valstep1)
    make_control('jitter',     0.05, 0, 0.25, STARTING_PARAMS['jitter'])

    make_control('tau_1',    0.40, 0, 10, STARTING_PARAMS['tau_1'], is_right=True, valstep=valstep1) 
    make_control('tau_2',    0.35, 0, 10, STARTING_PARAMS['tau_2'], is_right=True, valstep=valstep1)
    make_control('rho',      0.30, 0, 50, STARTING_PARAMS['rho'], is_right=True, valstep=valstep1)
    make_control('beta',     0.25, 1, 20, STARTING_PARAMS['beta'], is_right=True, valstep=valstep1)
    make_control('k',        0.20, -2, 20, STARTING_PARAMS['k'], is_right=True, valstep=valstep1)
    make_control('duration', 0.15, 10, 500, STARTING_PARAMS['duration'], is_right=True) 
    make_control('alpha',    0.10, 0.01, 5, STARTING_PARAMS['alpha'], is_right=True, valstep=valstep1) 

# Narrower range
if RANGE == 'narrow':
    make_control('c_ee',  0.40, 0, 5, STARTING_PARAMS['c_ee'], valstep=valstep1)
    make_control('c_ei',  0.35, -5, 0, STARTING_PARAMS['c_ei'], valstep=valstep1)
    make_control('c_ie',  0.30, 0, 5, STARTING_PARAMS['c_ie'], valstep=valstep1)
    make_control('c_ii',  0.25, -10, 0, STARTING_PARAMS['c_ii'], valstep=valstep1)
    make_control('P',     0.20, -5, 5, STARTING_PARAMS['P'], valstep=valstep1)
    make_control('Q',     0.15, -5, 5, STARTING_PARAMS['Q'], valstep=valstep1)
    make_control('jitter',     0.05, 0, 0.25, STARTING_PARAMS['jitter'])

    make_control('tau_1',    0.40, 0, 5, STARTING_PARAMS['tau_1'], is_right=True, valstep=valstep1) 
    make_control('tau_2',    0.35, 0, 5, STARTING_PARAMS['tau_2'], is_right=True, valstep=valstep1)
    make_control('rho',      0.30, 0, 35, STARTING_PARAMS['rho'], is_right=True, valstep=valstep1)
    make_control('beta',     0.25, 1, 20, STARTING_PARAMS['beta'], is_right=True)
    make_control('k',        0.20, 0, 10, STARTING_PARAMS['k'], is_right=True, valstep=valstep1)
    make_control('duration', 0.15, 10, 500, STARTING_PARAMS['duration'], is_right=True) 
    make_control('alpha',    0.10, 0.01, 5, STARTING_PARAMS['alpha'], is_right=True, valstep=valstep1) 

ax_radio = plt.axes([0.02, 0.6, 0.10, 0.2])
topologies = ('ring', 'line', 'lattice', 'full', 'smallworld')
radio_top = RadioButtons(ax_radio, topologies, active=topologies.index(STARTING_PARAMS['topology']))

s_N = make_control('N', 0.1, 2, 100, STARTING_PARAMS['N'])
s_N.valstep = 1


# ==========================================
# 6. UPDATE LOOP
# ==========================================
def update(val):
    global lines_E, lines_I, current_N, current_duration
    
    # 1. Grab values from UI
    N = int(s_N.val)
    duration = sliders['duration'].val
    
    # 2. Update Framework Config
    model_framework.config['model_params'].update({
        'c_ee': sliders['c_ee'].val,
        'c_ei': sliders['c_ei'].val,
        'c_ie': sliders['c_ie'].val,
        'c_ii': sliders['c_ii'].val,
        'P': sliders['P'].val,
        'Q': sliders['Q'].val,
        'tau_1': sliders['tau_1'].val,
        'tau_2': sliders['tau_2'].val,
        'rho': sliders['rho'].val,
        'beta': sliders['beta'].val,
        'k': sliders['k'].val,
        'alpha': sliders['alpha'].val
    })
    model_framework.config['network_params'].update({
        'topology': radio_top.value_selected,
        'N': N
    })
    model_framework.config['simulation']['duration'] = duration
    model_framework.config['simulation']['jitter'] = sliders['jitter'].val
    
    # 3. Run Native Framework Execution
    model_framework.create_network()
    model_framework.initialise_model()
    model_framework.run_simulation()
    
    # 4. Extract Results (Transpose to shape (steps, N) for plotting)
    t_array = model_framework.model.time_array
    E = model_framework.model.trajectories[0].T
    I = model_framework.model.trajectories[1].T

    # 5. Handle Plotting Updates
    if N != current_N or duration != current_duration or len(lines_E) == 0:
        ax.clear()
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, duration)
        ax.set_title(f"Metapopulation Native ({N}-Node {radio_top.value_selected})")
        ax.axvline(TRANSIENT_CUTOFF, color='red', linestyle='--', alpha=0.3)
        lines_E = ax.plot(t_array, E, color='blue', alpha=0.4, linewidth=1)
        lines_I = ax.plot(t_array, I, color='black', alpha=0.4, linewidth=1)
        current_N = N
        current_duration = duration
    else:
        for n in range(N):
            lines_E[n].set_ydata(E[:, n])
            lines_I[n].set_ydata(I[:, n])
            
    # 6. Handle Metrics Display
    if check_metrics.get_status()[0]:
        metrics = calculate_metrics(E, I, duration)
        txt_metrics.set_text(
            f"=== LIVE METRICS ===\n"
            f"Osc_Det:  {metrics['Oscillation_Detected']}\n"
            f"E_amp:    {metrics['E_amp']:.3f}\n"
            f"I_amp:    {metrics['I_amp']:.3f}\n"
            f"E_peaks:  {metrics['E_peaks']}\n"
            f"I_peaks:  {metrics['I_peaks']}\n"
            f"I:E Ratio:{metrics['Peak_Ratio_I_E']:.2f}\n"
            f"PLV_E:    {metrics['PLV_E']:.3f}\n"
        )
    else:
        txt_metrics.set_text("Metrics Disabled\n(Check box to enable)")
        
    fig.canvas.draw_idle()

radio_top.on_clicked(update)
check_metrics.on_clicked(update)

# Trigger the initial draw
update(None)
plt.show()