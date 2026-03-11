"""

Metapopulation class
    - a network of coupled DDE nodes

"""

import json

class Metapopulation():
    
    def __init__(self, config_name):
        
        with open(config_name) as f:
            self.config = json.load(f)
        
        # Set network paramaters    
        self.network_type = self.config["network_type"]  # Type of network, should match a function in networks.py
        self.N = self.config["network_params"]["N"]  # Number of nodes in network
        self.p = self.config["network_params"]["p"]
        self.A = None  # Adjacency matrix of network
        
        # Set delays
        self.delays = self.config["delays"]
        
        
        # Set simulation paramters
        self.T = self.config["simulation"]["T"]
        self.dt = self.config["simulation"]["dt"]
        self.n_runs = self.config["simulation"]["n_runs"]
        
        self.network_params = self.config["network_params"]
        self.delays = self.config["delays"]
        
        # Set model parameters
        self.model_name = self.config["model"]["name"]
        self.model_params = self.config["model"]["params"]
        
        
        def run_simulation():
            pass
        