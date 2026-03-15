
from metapopulation import Metapopulation
        
# Example run
model = Metapopulation()
model.load_config("config/example.yaml")
model.create_network()
model.network.plot_adjacency_matrix()
model.initialise_model()
model.run_simulation(duration=1000, dt=0.1)
model.plot_trajectories()
        