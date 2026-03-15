# The role of network structure and time delay in a metapopulation Wilson–Cowan model, Conti and Van Gorder, 2019

## Directory structure
- config: configuration files for the Metapopulation class
- data: a place for data to live
- ref: any references, notes, etc
- scrap: a messy playground
- src: where all the good code goes

## Files
- ReadMe.md - add anything a contributor needs to know here
- requirements.txt - list of python packages required to run code

## Code files
- models.py - any DDE models we implement, e.g. Wilson-Cowan
- networks.py - network generators and analysers
- utilities.py - useful bits and pieces
- report_plots.py - one place for all plots, where we can fix themes and formatting
- metapopulation.py - class to generate and manipulate a complete DDE coupled network model

## Setup
- Create virtual environment

```python -m venv .venv```

- Install packages from requirements.txt

```pip install -r requirements.txt```

- Run file example.py for example

```
model = Metapopulation()  # Initialise the model
model.load_config("config/example.yaml")  # Load a configuration file - create your own to play with parameters
model.create_network()  # Create the network. Edit config or pass a dictionary of different parameters
model.network.plot_adjacency_matrix()  # Plot the network connectivity matrix
model.initialise_model()  # Set up the wilson-cowan model (not yet implemented) - Edit config or pass a dictionary of different parameters
model.run_simulation(duration=1000, dt=0.1)  # Run the simulation - generates trajectories
model.plot_trajectories()  # Plot the Excitatory (blue) and Inhibitory (black) trajectories for each node
```

## Reporting

- Link: https://www.overleaf.com/8621396969mrhntkbkqvxb#fa30fc
