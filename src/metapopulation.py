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
import matplotlib.pyplot as plt
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
            
    def run_simulation(self, duration=1000, dt=0.1):
        """ Runs n simulations with the specified model
            Results are stored in self.trajectories
            Time grid is stored in self.time_array
        """
        # Initialise model simulation
        self.model.set_time_grid(duration, dt)
        self.model.set_initial_conditions(self.network.N)
        
        # Run simulation
        self.model.run() 
              
        
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