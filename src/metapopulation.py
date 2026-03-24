"""

Metapopulation class
    - Models a network of coupled DDE nodes
    - Create network ,each node corresponds to a neural population (create_network)
    - Set model and parameters, e.g. Wilson-Cowan (initialise_model)
    - Run model simulation to calculate node trajectories (run_simulation)
    - Simple plot all trajectories function, replicates figure 4. plots (plot_trajectories)
"""

import yaml
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.network import Network
from src.model import Model

class Metapopulation():
    
    def __init__(self):
        pass
        
    def load_config(self, config_name):
        """ Load experiment from config file """
        with open(config_name) as f:
            self.config = yaml.safe_load(f)
            
    def create_network(self, params=None):
        """ Creates network from config file parameters
            You can change the network parameters on the fly by passing them in a dictionary, 
            e.g. {'node_distance': 0.5}
        """
        # Update network parameters
        network_params = self.config['network_params']
        if params is not None:
            for param, val in params.items():
                if param in network_params.keys():
                    network_params[param] = val
                
        # Create network
        self.network = Network(network_params)       
        
    def initialise_model(self, params=None):
        """ Set up the model and modelling parameters"""
        # Update model parameters
        model_params = self.config['model_params']
        if params is not None:
            for param, val in params.items():
                if param in model_params.keys():
                    model_params[param] = val
        
        # Initialise model
        self.model = Model(
            self.network,
            params=self.config['model_params']
        )
            
    def run_simulation(self, duration=1000, dt=0.1, initial_conditions=None, timeit=False):
        """ Runs n simulations with the specified model
            Results are stored in self.trajectories
            Time grid is stored in self.time_array
        """
        if timeit:
            t0 = time.time()
            print("Running simulation with parameters:")
            print(self.config)
        
        # Update simulation parameters from config and arguments
        duration = self.config['simulation'].get('duration', duration)
        dt = self.config['simulation'].get('dt', dt)
        initial_conditions = self.config['simulation'].get('initial_conditions', initial_conditions)
        
        # Initialise model simulation
        self.model.set_time_grid(duration, dt)
        self.model.set_initial_conditions(initial_conditions)
        
        # Run simulation
        self.model.run() 
        
        # Oprional time the run
        if timeit:
            print(f"Simulation runtime: {(time.time() - t0):.2f} seconds")
              
        
    def plot_trajectories(self):
        """ Plots the trajectories from simulation
            Select specific indices to plot
        """
        for i, c in enumerate(self.model.components):
            t = self.model.time_array
            y = self.model.trajectories[i]
            colour = 'blue' if c == 'Excitatory' else 'black'
            plt.plot(t, y.T, color=colour, alpha=0.5, linewidth=1)
            # Add single legend item
            plt.plot(0, 0, color=colour, label=c)      
        plt.legend()
        plt.show()      
        
        
    def plot_nullclines(self):
        """
        Plots the E and I nullclines for ALL nodes at t=0.
        """
        
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-p['beta'] * x))
        
        # Get parameters and values
        p = self.model.params
        N = self.network.N
        W = self.network.A
        E_init = self.model.initial_conditions[:N]
        I_init = self.model.initial_conditions[N:]
        coupling_terms = p['k'] * (W @ E_init)
        
        # Set up the plot
        E_val = np.linspace(0.0, 1.1, 100)
        I_val = np.linspace(0.0, 1.1, 100)
        E_grid, I_grid = np.meshgrid(E_val, I_val)
        plt.figure(figsize=(8, 6))
        
        for node in range(N):
            
            # Calculate the derivative
            dE = -E_grid + sigmoid(p['c_ee']*E_grid + p['c_ie']*I_grid + p['P'] + p['k'] * coupling_terms[node])
            dI = -I_grid + sigmoid(p['c_ei']*E_grid + p['c_ii']*I_grid + p['Q'])
            
            # Plot the 0 contours, i.e. where dE=0 and dI=0
            plt.contour(E_grid, I_grid, dE, levels=[0], colors='blue', linewidths=2, alpha=0.7)  # I nullcline
            plt.contour(E_grid, I_grid, dI, levels=[0], colors='black', linewidths=2, alpha=0.7)  # E nullcline
            plt.plot(E_init[node], I_init[node], 'ro', markersize=4, alpha=0.7)  # Initial conditions
        
        custom_lines = [Line2D([0], [0], color='blue', lw=2, alpha=0.7),
                        Line2D([0], [0], color='black', lw=2, alpha=0.7),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6)]
        plt.xlabel("Excitatory")
        plt.ylabel("Inhibitory")
        plt.legend(custom_lines, ['E-Nullclines', 'I-Nullclines', 'Initial Conditions'])
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.show()
