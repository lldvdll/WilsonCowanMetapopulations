"""
Example usage of main class Metapopulation
    - Instantiates class
    - Loads example config file
    - Creates network using config file parameter
    - Initialises the model using config file parameters
    - Runs simulation - 1000ms, 0.1ms timestep
    - Plots trajectories
    
Note: you can edit network or model parametes by running create_network or initialise_model and passing a dictionary of the parameters you want to change

"""

import os
from src.metapopulation import Metapopulation
        
# Example run
model = Metapopulation()
model.load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "example.yaml"))  # Specify your config file here. It should sit in experiments folder, same name as this file
model.create_network()  # Create the network. You can add a dictionary of parameters here
model.network.plot_adjacency_matrix()  # Plot the network connectivity matrix 
model.initialise_model()  # Initialise the model, sets parameters from config. You can also add a dictionary of parameters here
model.run_simulation(duration=1000, dt=0.1)  # Run simulation specifying time grid
model.plot_trajectories()  # Plot the Excitatory (blue) and Inhibitory (black) trajectories for each node
        