"""

Metapopulation class
    - a network of coupled DDE nodes

"""

import yaml

from network import Network

class Metapopulation():
    
    def __init__(self, config_name):
        
        with open(config_name) as f:
            self.config = yaml.safe_load(f)
            
        print(self.config)
            
        self.network = Network(self.config["network"])
        
        
        def run_simulation():
            pass
        
        
data = Metapopulation("config/example.yaml")
data.network.plot_network_matrix('Adjacency')
data.network.plot_network_matrix('Delay')
        