"""
Reproduce  figure 3 from Conti & Van Gorder (2018) using model.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.metapopulation import Metapopulation

def plot_trajectory_on_ax(ax, title, model):
    """Draws the trajectory on a explicitly targeted matplotlib axis."""
    t = model.model.time_array
    E = model.model.trajectories[0]
    I = model.model.trajectories[1]
    
    ax.plot(t, E.T, color='blue', alpha=0.2, linewidth=1)
    ax.plot(t, I.T, color='black', alpha=0.2, linewidth=1)
    
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)


def load_config_from_csv(path, rows):
    df = pd.read_csv(path)
    df = df.iloc[[r-2 for r in rows], 3:18]
    print(df)
    return df

def run_simulations(base_config_path, model_configs):
    models = []
    for i, model_config in model_configs.iterrows():
        model = Metapopulation()
        model.load_config(base_config_path)
        print(model_config.to_dict())
        network_params = {
            'topology': model_config['topology'],
            'N': int(model_config['N']),
            'p': None,
            'normalise': False
        }
        model.create_network(network_params)         
        model.initialise_model(model_config)
        if 'jitter' in model_config:
            model.config['simulation']['jitter'] = model_config['jitter']
        if 'duration' in model_config:
            model.config['simulation']['duration'] = model_config['duration']
        model.run_simulation()
        models.append(model)
    return models

def format_param_title(param_dict):
    """Converts a parameter dict into a wrapped string with MathText symbols."""
    # Map long variable names to Matplotlib MathText symbols
    symbols = {
        'c_ee': '$c_{ee}$', 'c_ei': '$c_{ei}$', 'c_ie': '$c_{ie}$', 'c_ii': '$c_{ii}$',
        'tau_1': '$\\tau_1$', 'tau_2': '$\\tau_2$', 'rho': '$\\rho$', 'beta': '$\\beta$',
        'alpha': '$\\alpha$', 'P': '$P$', 'Q': '$Q$', 'k': '$k$'
    }
    
    parts = []
    for key, val in param_dict.items():
        if key in ['N', 'duration']:
            continue
        sym = symbols.get(key, key)
        # Format numbers to drop trailing zeros (e.g., 10.0 -> 10)
        val_str = f"{val:g}" if isinstance(val, (float, int)) else str(val)
        parts.append(f"{sym}={val_str}")
        
    # Group into chunks of 5 items per line to wrap nicely
    chunk_size = 6
    lines = [", ".join(parts[i:i + chunk_size]) for i in range(0, len(parts), chunk_size)]
    
    return "\n".join(lines)
        


def main():
    
    # Paths
    script_path = os.path.dirname(os.path.realpath(__file__))
    csv_config_path = os.path.join(script_path, 'parameter_reference.csv')
    base_config_path = os.path.join(script_path, 'network_effects.yaml')
    
    # Load config from csv
    rows = [19]
    model_configs = load_config_from_csv(csv_config_path, rows)
    topologies = ['ring', 'line', 'lattice', 'full']
    plot_titles = ['Cycle', 'Path', 'Lattice', 'Complete']
    model_configs = pd.concat([model_configs] * 4, ignore_index=True)
    model_configs['topology'] = topologies
    print(model_configs)
    models = run_simulations(base_config_path, model_configs)
    
    # Set up plot
    nrows = 4
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle("Network Topology Effects", fontsize=16)
    
    # Get parameter strings
    param_strs = [
        format_param_title(model_configs.iloc[i,:-1].to_dict())
        for i in range(2)
    ]
    
    # Plot 4 network topologies
    plot_trajectory_on_ax(ax[0, 0], 'Cycle', models[0])
    plot_trajectory_on_ax(ax[0, 1], 'Path', models[1])
    plot_trajectory_on_ax(ax[1, 0], 'Lattice', models[2])
    plot_trajectory_on_ax(ax[1, 1], 'Complete', models[3])

    # Formatting Trajectories
    # ax[0, 0].set_ylabel(f"Noise = {int(j1*100)}%")
    # ax[1, 0].set_ylabel(f"Noise = {int(j2*100)}%")
    # ax[2, 0].set_ylabel(f"k = {k}")
    # ax[3, 0].set_ylabel(f"'$\\rho$' = {rho}")
    for i in range(2):
        ax[1, i].set_xlabel("t")
    
    plt.tight_layout()
    filepath = os.path.join(script_path, 'network_effects.png')
    plt.savefig(filepath, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()