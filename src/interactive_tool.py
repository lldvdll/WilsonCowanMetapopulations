"""
Advanced Real-time Interactive Wilson-Cowan DDE Simulator
Features: Topology, Symmetry Breaking, Linked TextBoxes, Alpha Slider, and Save/Load Presets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, TextBox, Button
import networkx as nx
import json
import os

# 1. Base Simulation Setup
dt = 0.1
current_N = 16
current_duration = 100
PARAMS_FILE = 'saved_params.json'
preset_index = 0

# 2. Plotting Setup
fig, ax = plt.subplots(figsize=(15, 10))
plt.subplots_adjust(left=0.25, bottom=0.5, right=0.95) 
ax.set_title("Vectorized N-Node Wilson-Cowan DDE Simulator")
ax.set_xlabel("Time")
ax.set_ylabel("Activity")
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(0, current_duration)
ax.grid(True, alpha=0.3)

lines_E = []
lines_I = []

# 3. Custom UI Controls (Slider + Linked TextBox)
sliders = {}
def make_control(name, y_pos, min_val, max_val, init_val, is_right=False):
    x_base = 0.60 if is_right else 0.25
    ax_slider = plt.axes([x_base, y_pos, 0.15, 0.02])
    ax_text = plt.axes([x_base + 0.16, y_pos, 0.05, 0.02])
    
    sl = Slider(ax_slider, name, min_val, max_val, valinit=init_val)
    sl.valtext.set_visible(False)
    
    txt = TextBox(ax_text, '', initial=f"{init_val:.2f}")
    
    def on_text_submit(text):
        try:
            val = float(text)
            sl.set_val(val) 
        except ValueError:
            txt.set_val(f"{sl.val:.2f}") 
            
    def on_slider_change(val):
        txt.set_val(f"{val:.2f}")
        update(None) 
        
    txt.on_submit(on_text_submit)
    sl.on_changed(on_slider_change)
    
    sliders[name] = sl
    return sl

# Left Column (Weights, Stimuli, Noise)
make_control('c_ee',  0.40, -25, 25, 1.00)
make_control('c_ei',  0.35, -25, 25, -6.42)
make_control('c_ie',  0.30, -25, 25, 15.45)
make_control('c_ii',  0.25, -25, 25, -14.00)
make_control('P',     0.20, -10, 10, -3.54)
make_control('Q',     0.15, -10, 10, 5.00)
make_control('noise', 0.10, 0, 0.1, 0.02) 

# Right Column (Delays, Globals, Duration, Alpha)
make_control('tau_1',    0.40, -10, 50, 0.30, is_right=True) 
make_control('tau_2',    0.35, -10, 50, 1.77, is_right=True)
make_control('rho',      0.30, -10, 50, 10.00, is_right=True)
make_control('beta',     0.25, 1, 20, 10.00, is_right=True)
make_control('k',        0.20, -2, 20, 1.20, is_right=True)
make_control('duration', 0.15, 10, 500, 100.0, is_right=True) 
make_control('alpha',    0.10, 0.01, 5, 0.60, is_right=True) # NEW ALPHA SLIDER

# Far Left Panel (Network Controls)
ax_radio = plt.axes([0.02, 0.6, 0.15, 0.2])
radio_top = RadioButtons(ax_radio, ('ring', 'line', 'lattice', 'full', 'smallworld'))

ax_N = plt.axes([0.05, 0.5, 0.1, 0.02])
ax_N_txt = plt.axes([0.16, 0.5, 0.04, 0.02])
s_N = Slider(ax_N, 'N', 2, 30, valinit=16, valstep=1)
s_N.valtext.set_visible(False)
txt_N = TextBox(ax_N_txt, '', initial='16')

def on_n_submit(text):
    try: s_N.set_val(int(text))
    except: txt_N.set_val(str(int(s_N.val)))
def on_n_change(val):
    txt_N.set_val(str(int(val)))
    update(None)

# ---------------------------------------------------------
# PRESET SAVE/LOAD LOGIC
# ---------------------------------------------------------
def load_presets():
    if not os.path.exists(PARAMS_FILE):
        default_data = [
            {"name": "Complex Bursting", "N": 16, "topology": "ring", "c_ee": 1.0, "c_ei": -6.42, "c_ie": 15.45, "c_ii": -14.0, "P": -3.54, "Q": 5.0, "tau_1": 0.3, "tau_2": 1.77, "rho": 10.0, "beta": 10.0, "k": 1.2, "noise": 0.02, "duration": 100.0, "alpha": 0.6},
            {"name": "Fig 2b Flatline Base", "N": 2, "topology": "ring", "c_ee": 1.0, "c_ei": -1.0, "c_ie": 1.0, "c_ii": -1.0, "P": 0.5, "Q": 0.5, "tau_1": 4.0, "tau_2": 40.0, "rho": 10.0, "beta": 10.0, "k": 1.0, "noise": 0.0, "duration": 100.0, "alpha": 0.6}
        ]
        with open(PARAMS_FILE, 'w') as f:
            json.dump(default_data, f, indent=4)
        return default_data
    with open(PARAMS_FILE, 'r') as f:
        return json.load(f)

def save_current_state(event):
    presets = load_presets()
    new_preset = {
        "name": f"Saved Preset {len(presets) + 1}",
        "N": int(s_N.val),
        "topology": radio_top.value_selected
    }
    for key, sl in sliders.items():
        new_preset[key] = round(sl.val, 3)
    presets.append(new_preset)
    with open(PARAMS_FILE, 'w') as f:
        json.dump(presets, f, indent=4)
    print(f"Saved to {PARAMS_FILE}!")

def cycle_preset(event):
    global preset_index
    presets = load_presets()
    if not presets: return
    
    preset_index = (preset_index + 1) % len(presets)
    p = presets[preset_index]
    
    print(f"Loading Preset: {p['name']}")
    s_N.set_val(p.get("N", 16))
    radio_top.set_active(["ring", "line", "lattice", "full", "smallworld"].index(p.get("topology", "ring")))
    for key, sl in sliders.items():
        if key in p: sl.set_val(p[key])

# Buttons UI
ax_save = plt.axes([0.80, 0.05, 0.07, 0.04])
ax_next = plt.axes([0.88, 0.05, 0.09, 0.04])
btn_save = Button(ax_save, 'Save Params')
btn_next = Button(ax_next, 'Next Saved')
btn_save.on_clicked(save_current_state)
btn_next.on_clicked(cycle_preset)

# 4. Network Generation
def get_normalized_matrix(N, topology):
    N = int(N)
    if topology == 'line': G = nx.path_graph(N)
    elif topology == 'lattice':
        side = int(np.ceil(np.sqrt(N)))
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
        G.remove_nodes_from(list(G.nodes())[N:])
    elif topology == 'full': G = nx.complete_graph(N)
    elif topology == 'smallworld': G = nx.watts_strogatz_graph(N, max(2, int(0.2*N)), 0.1)
    else: G = nx.circulant_graph(N, [i+1 for i in range(max(1, int(0.2*N)))])
        
    W = nx.to_numpy_array(G)
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return W / row_sums[:, np.newaxis]

# 5. Core Vectorized Solver
def update(val):
    global lines_E, lines_I, current_N, current_duration
    
    N = int(s_N.val)
    duration = sliders['duration'].val
    noise_level = sliders['noise'].val
    
    W = get_normalized_matrix(N, radio_top.value_selected)
    
    steps = int(duration / dt)
    t_array = np.linspace(0, duration, steps)
    
    E = np.full((steps, N), 0.25)
    I = np.full((steps, N), 0.75)
    
    t1 = max(1, int(sliders['tau_1'].val / dt))
    t2 = max(1, int(sliders['tau_2'].val / dt))
    tr = max(1, int(sliders['rho'].val / dt))
    max_delay = max(t1, t2, tr)
    
    np.random.seed(42) 
    E[:max_delay, :] += np.random.uniform(-noise_level, noise_level, N)
    I[:max_delay, :] += np.random.uniform(-noise_level, noise_level, N)
    
    c_ee = sliders['c_ee'].val; c_ei = sliders['c_ei'].val
    c_ie = sliders['c_ie'].val; c_ii = sliders['c_ii'].val
    P = sliders['P'].val; Q = sliders['Q'].val
    beta = sliders['beta'].val; k = sliders['k'].val
    alpha = sliders['alpha'].val 

    for i in range(max_delay, steps - 1):
        coupling = W @ E[i - tr] 
        e_arg = c_ee * E[i - t1] + c_ie * I[i - t2] + P + k * coupling
        i_arg = c_ei * E[i - t2] + c_ii * I[i - t1] + Q
        
        dE = -E[i] + 1.0 / (1.0 + np.exp(-beta * np.clip(e_arg, -10, 10)))
        
        # CORRECTED ALPHA INTEGRATION
        dI = alpha * (-I[i] + 1.0 / (1.0 + np.exp(-beta * np.clip(i_arg, -10, 10))))
        
        E[i+1] = E[i] + dE * dt
        I[i+1] = I[i] + dI * dt

    if N != current_N or duration != current_duration or len(lines_E) == 0:
        ax.clear()
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, duration)
        ax.set_xlabel("Time")
        ax.set_ylabel("Activity")
        ax.set_title(f"Vectorized {N}-Node Wilson-Cowan DDE Simulator")
        ax.grid(True, alpha=0.3)
        
        lines_E = ax.plot(t_array, E, color='blue', alpha=0.4, linewidth=1)
        lines_I = ax.plot(t_array, I, color='black', alpha=0.4, linewidth=1)
        ax.plot([], [], 'b-', label='Excitatory (E)')
        ax.plot([], [], 'k-', label='Inhibitory (I)')
        ax.legend(loc='upper right')
        
        current_N = N
        current_duration = duration
    else:
        for n in range(N):
            lines_E[n].set_ydata(E[:, n])
            lines_I[n].set_ydata(I[:, n])
            
    fig.canvas.draw_idle()

txt_N.on_submit(on_n_submit)
s_N.on_changed(on_n_change)
radio_top.on_clicked(update)

update(None)
plt.show()