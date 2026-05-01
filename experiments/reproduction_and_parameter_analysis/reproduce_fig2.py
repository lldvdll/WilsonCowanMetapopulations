"""
Reproduce  figure 2 from Conti & Van Gorder (2018) using metapopulation.py.

Fig. 2. Numerical solutions for the 2-node Wilson–Cowan model. 
Resulting dynamics are shown for 
    (a) τn,1 = 1, τn,2 = 1.4, 
    (b) τn,1 = 4, τn,2 = 40. 
Other parameter values are cee,n = cei,n = 1, cie,n = cii,n = −1, kn = 1, and Pn = Qn = 0.5, for n = 1, 2, while ρ1,2 = 10. 
    In (c) we instead show a solution curve corresponding to the chaotic parameter regime cee,n = cii,n = −6, cei,n = cie,n = 2.5, Pn = Qn = 0.2, τn,1 = 1, τn,2 = 1.4, k1 = k2 = 11. 
The blue solid curves correspond to the activities of the excitatory subpopulations (E), while the black dashed curves correspond to the activities of the inhibitory subpopulations (I)

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
    
    ax.plot(t, E.T, color='blue', alpha=0.5, linewidth=1)
    ax.plot(t, I.T, color='black', alpha=0.5, linewidth=1)
    
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

def plot_nullcline_on_ax(ax, title, model):
    """Draws the nullclines on an explicitly targeted matplotlib axis."""
    p = model.model.params
    N = model.network.N
    W = model.network.A
    E_init = model.model.initial_conditions[:N]
    I_init = model.model.initial_conditions[N:]
    coupling_terms = p['k'] * (W @ E_init)
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-p['beta'] * x))
    
    E_val = np.linspace(0.0, 1.1, 100)
    I_val = np.linspace(0.0, 1.1, 100)
    E_grid, I_grid = np.meshgrid(E_val, I_val)
    
    for node in range(N):
        dE = -E_grid + sigmoid(p['c_ee']*E_grid + p['c_ie']*I_grid + p['P'] + p['k'] * coupling_terms[node])
        dI = -I_grid + sigmoid(p['c_ei']*E_grid + p['c_ii']*I_grid + p['Q'])
        
        ax.contour(E_grid, I_grid, dE, levels=[0], colors='blue', linewidths=2, alpha=0.5)
        ax.contour(E_grid, I_grid, dI, levels=[0], colors='black', linewidths=2, alpha=0.5)
        ax.plot(E_init[node], I_init[node], 'ro', markersize=4, alpha=0.7)
        
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)


def load_config_from_csv(path, rows):
    df = pd.read_csv(path)
    df = df.iloc[[r-2 for r in rows], 5:17]
    return df

def run_simulations(base_config_path, model_configs):
    models = []
    for i, model_config in model_configs.iterrows():
        model = Metapopulation()
        model.load_config(base_config_path)
        model.create_network()
        model.initialise_model(model_config)
        model.run_simulation()
        models.append(model)
    return models
        


def main():
    
    script_path = os.path.dirname(os.path.realpath(__file__))
    
    # Load config from csv
    csv_config_path = os.path.join(script_path, 'parameter_reference.csv')
    rows = [3, 4, 5, 6, 7, 8]
    model_configs = load_config_from_csv(csv_config_path, rows)
    
    # Run simulations
    base_config_path = os.path.join(script_path, 'reproduce_fig2.yaml')
    models = run_simulations(base_config_path, model_configs)
    
    
    # Set up plot

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    # fig.suptitle("Reproduction of Fig. 2", fontsize=16)
    
    # EXPLICITLY Map specific models to specific 2D coordinates
    plot_trajectory_on_ax(ax[0, 0], "(a)", models[0])
    plot_trajectory_on_ax(ax[0, 1], "(b)", models[2])
    plot_trajectory_on_ax(ax[0, 2], "(c)", models[4])
    
    plot_trajectory_on_ax(ax[1, 0], "", models[1])
    plot_trajectory_on_ax(ax[1, 1], "", models[3])
    plot_trajectory_on_ax(ax[1, 2], "", models[5])

    # Formatting Trajectories
    ax[0, 0].set_ylabel("Listed parameters")
    ax[1, 0].set_ylabel("Corrected parameters")
    for i in range(3):
        ax[1, i].set_xlabel("t")
    
    plt.tight_layout()
    filepath = os.path.join(script_path, 'reproduce_fig2.png')
    plt.savefig(filepath, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()