# The role of network structure and time delay in a metapopulation Wilson–Cowan model, Conti and Van Gorder, 2019

## Directory structure
- data: a place for data to live
- experiments: Experiments using the Metapopulation class. Python script and accompanying .yaml config file
- ref: any references, notes, etc
- scrap: a messy playground. Anything which doesn't fit structure belongs here. Commit it if it's useful
- src: where all the good code goes. the Metapopulation package

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

- Install the src package with pip 

```pip install -e .```

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

## Next-Generation Neural Mass Model (NG NMM) track — Ningqian

This track extends the project to the Next-Generation Neural Mass Model
of Forrester et al. (2024), an exact mean-field reduction of a QIF
neuron population obtained via the Ott–Antonsen ansatz. The NG NMM is
applied to the HCP DK68 connectome to model BOLD-FC under
γ-distributed conduction-velocity delays and with electrical
(gap-junction) coupling.

### Code

- `src/next_gen_model.py` — single-node NG NMM (12 ODEs per region;
  scipy `solve_ivp`).
- `src/next_gen_network.py` — N-region network NG NMM
  (`12N + 2M` DDE state variables, jitcdde-compiled). Provides
  Kuramoto Z, PLV, BOLD via Balloon–Windkessel, and helpers for
  fitting against empirical FC.

### Experiments

The three reported sweeps (operating point: $\eta_E = -2.15$
super-critical, $s = 1.4$, $v_m = 12$ m/s,
$\kappa_v$ scaled from Forrester defaults; γ-velocity delays):

- **`experiments/c3_gamma_vm_sweep.py`** — velocity sensitivity sweep
  $v_m \in \{6, 8, 10, 12, 15\}$ m/s. 

- **`experiments/c3_kappa_v_sweep_HCP.py`** — gap-junction sweep
  $\kappa_v$-scale $\in \{0, 0.5, 1, 2, 3, 4, 5, 6\}$ on the Forrester
  defaults ($\kappa_v^{EE} = 0.01$, $\kappa_v^{II} = 0.025$). 

- **`experiments/c6_scfc_supercritical.py`** — structural baseline at
  the operating point. 

### Setup specifics for this track

1. The standard `pip install -r requirements.txt` covers everything,
   but two dependencies are unusual:
   - **`enigmatoolbox`** is **not on PyPI**. The
     `requirements.txt` line installs it directly from GitHub
     (`pip install git+https://github.com/MICA-MNI/ENIGMA.git`).
   - **`abagen`** is on PyPI but ships its own DK68 volumetric atlas;
     first use will trigger a one-time data download under
     `~/abagen-data/`.

2. **Recommended Python environment:** Anaconda (or any Python 3.10+
   that supports the `vtk` wheel — required by enigmatoolbox).

3. **Data preparation (run once before any experiments):**

   ```bash
   python data/prepare_data.py
   ```

   This downloads the HCP DK68 SC and rs-fMRI FC matrices from the
   ENIGMA Toolbox (Larivière et al. 2021, Nat Methods), computes
   region centroids from abagen's volumetric DK68 atlas (Markello
   et al. 2021, eLife), and saves five `.npy` files to `data/`:

   - `hcp_sc_68.npy`        — group-averaged SC (207 HCP subjects)
   - `hcp_fc_68.npy`        — group-averaged Fisher-$z$ rs-fMRI FC
   - `hcp_labels_68.npy`    — DK68 region labels
   - `hcp_centroids_68.npy` — abagen volumetric centroids (mm)
   - `hcp_dist_68.npy`      — pairwise Euclidean distances (mm)
