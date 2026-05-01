"""
Test that setting p to None in config returns a network where only immediate neighbours are connected

"""

import os
import time
from src.metapopulation import Metapopulation
        
# Example run
model = Metapopulation()
model.load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_ring_p_none.yaml"))
model.create_network()
model.network.plot_adjacency_matrix()