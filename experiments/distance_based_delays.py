"""
Distance-Based Delays Experiment for Wilson-Cowan Metapopulation Model

This script extends the basic Wilson-Cowan model to use DISTANCE-BASED inter-node
delays instead of constant delays. 

Key Features:
    1. Assigns spatial coordinates to network nodes based on topology
    2. Computes Euclidean distances between nodes
    3. Converts distances to time delays: delay = base + distance/velocity
    4. Integrates with existing Metapopulation class seamlessly

Usage:
    python experiments/distance_based_delays.py

To modify delay parameters:
    - Edit distance_based_delays.yaml configuration file
    - Or pass parameters directly to DistanceDelayedMetapopulation class

The script will:
    - Load configuration
    - Create network with spatial positions
    - Compute distance-based delay matrix
    - Run Wilson-Cowan simulation
    - Visualize results (trajectories, network layout, delay matrix)

Author: Your Name
Date: 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metapopulation import Metapopulation
from src.distance_delays import (
    assign_node_coordinates,
    compute_delay_matrix,
    visualize_network_with_delays
)


class DistanceDelayedMetapopulation(Metapopulation):
    """
    Extension of Metapopulation class to support distance-based inter-node delays.
    
    This class inherits from Metapopulation and overrides the model initialization
    to inject a distance-based delay matrix instead of using a constant delay.
    
    The delay matrix is computed from:
        1. Node spatial positions (based on network topology)
        2. Euclidean distances between nodes
        3. Signal propagation velocity
        4. Optional baseline and maximum delays
    
    Attributes (in addition to Metapopulation attributes)
    ---------------------------------------------------
    node_positions : dict
        Spatial (x, y) coordinates for each node
    distance_matrix : np.ndarray
        Euclidean distances between all node pairs (N x N)
    delay_matrix : np.ndarray
        Time delays between all node pairs (N x N)
        This replaces the constant 'rho' parameter
    """
    
    def __init__(self):
        """Initialize the distance-delayed metapopulation model."""
        super().__init__()
        self.node_positions = None
        self.distance_matrix = None
        self.delay_matrix = None
        
    def create_network_with_positions(self, params=None):
        """
        Create network and assign spatial positions to nodes.
        
        This extends the base create_network() method to also compute
        spatial positions for distance-based delay calculations.
        
        Parameters
        ----------
        params : dict, optional
            Override network parameters from config
            
        Notes
        -----
        After calling this method:
            - self.network contains the graph structure (adjacency matrix)
            - self.node_positions contains (x, y) coords for each node
            
        Spatial positions are determined by network topology:
            - 'line': nodes along x-axis
            - 'ring': nodes on a circle  
            - 'lattice': nodes on 2D grid
            - 'full'/'smallworld': force-directed layout
        """
        # First, create the network structure using parent class method
        self.create_network(params=params)
        
        # Extract distance delay parameters from config
        dd_params = self.config.get('distance_delay_params', {})
        scale = dd_params.get('scale', 1.0)
        layout_seed = dd_params.get('layout_seed', 42)
        topology = self.network.config['topology']
        
        # Assign spatial positions based on topology
        print(f"\nAssigning spatial positions for {topology} topology...")
        print(f"  - Spatial scale: {scale}")
        
        self.node_positions = assign_node_coordinates(
            self.network,
            topology=topology,
            scale=scale,
            seed=layout_seed
        )
        
        print(f"  - Assigned positions to {len(self.node_positions)} nodes")
        
        # Print sample positions for verification
        if len(self.node_positions) <= 5:
            print(f"  - Node positions: {self.node_positions}")
        else:
            print(f"  - Sample positions: Node 0={self.node_positions[0]}, "
                  f"Node 1={self.node_positions[1]}")
    
    def compute_delay_matrix(self):
        """
        Compute the distance-based delay matrix from node positions.
        
        This method:
            1. Computes Euclidean distances from positions
            2. Converts distances to delays using signal velocity
            3. Applies baseline delay, max delay cap, adjacency mask
            4. Stores result in self.delay_matrix
        
        The delay formula is:
            delay[i,j] = base_delay + (distance[i,j] / signal_velocity)
        
        Notes
        -----
        This must be called AFTER create_network_with_positions()
        
        The resulting delay matrix will be injected into the Model
        when initialise_model() is called.
        
        Raises
        ------
        ValueError
            If node positions haven't been assigned yet
        """
        if self.node_positions is None:
            raise ValueError("Must call create_network_with_positions() first!")
        
        # Extract delay parameters from config
        dd_params = self.config.get('distance_delay_params', {})
        signal_velocity = dd_params.get('signal_velocity', 1.0)
        base_delay = dd_params.get('base_delay', 0.0)
        self_delay = dd_params.get('self_delay', 0.0)
        max_delay = dd_params.get('max_delay', None)
        
        print("\nComputing distance-based delay matrix...")
        print(f"  - Signal velocity: {signal_velocity} [spatial units / time units]")
        print(f"  - Base delay: {base_delay} [time units]")
        print(f"  - Self delay: {self_delay} [time units]")
        print(f"  - Max delay cap: {max_delay if max_delay else 'None (unlimited)'}")
        
        # Compute the delay matrix
        self.delay_matrix = compute_delay_matrix(
            positions=self.node_positions,
            adjacency_matrix=self.network.A,
            signal_velocity=signal_velocity,
            base_delay=base_delay,
            self_delay=self_delay,
            max_delay=max_delay
        )
        
        # Report statistics
        # Exclude diagonal (self-delays) and zeros (unconnected) from stats
        non_self_delays = self.delay_matrix[~np.eye(self.delay_matrix.shape[0], dtype=bool)]
        connected_delays = non_self_delays[non_self_delays > 0]
        
        if len(connected_delays) > 0:
            print(f"\nDelay matrix statistics:")
            print(f"  - Min inter-node delay: {connected_delays.min():.3f}")
            print(f"  - Max inter-node delay: {connected_delays.max():.3f}")
            print(f"  - Mean inter-node delay: {connected_delays.mean():.3f}")
            print(f"  - Number of connections: {len(connected_delays)}")
        else:
            print("\n  Warning: No connected node pairs found!")
        
        return self.delay_matrix
    
    def initialise_model_with_delays(self, params=None):
        """
        Initialize the Wilson-Cowan model with distance-based delays.
        
        This method extends the parent class by:
            1. Replacing the constant 'rho' parameter with the delay matrix
            2. Modifying the model construction to use heterogeneous delays
        
        Parameters
        ----------
        params : dict, optional
            Override model parameters from config
            
        Notes
        -----
        Currently, this method uses the computed delay_matrix, but the
        Model class (model.py) still needs to be modified to accept
        a MATRIX of delays instead of a single 'rho' value.
        
        TODO: Modify model.py to support heterogeneous delays
        For now, we'll use the MEAN delay as a single value
        
        Raises
        ------
        ValueError
            If delay matrix hasn't been computed yet
        """
        if self.delay_matrix is None:
            raise ValueError("Must call compute_delay_matrix() first!")
        
        # Check if distance delays are enabled
        dd_params = self.config.get('distance_delay_params', {})
        enabled = dd_params.get('enabled', True)
        
        if not enabled:
            print("\n⚠ Distance-based delays DISABLED in config.")
            print("  Using constant 'rho' from model_params instead.")
            # Use parent class method with constant delays
            self.initialise_model(params=params)
            return
        
        # ======================================================================
        # IMPORTANT NOTE:
        # ======================================================================
        # The current Model class (model.py) uses a SINGLE delay value 'rho'
        # for all inter-node connections. To fully implement distance-based
        # delays, model.py needs to be modified to accept a DELAY MATRIX.
        #
        # For now, we have two options:
        #   Option 1: Use the MEAN delay (what we do below)
        #   Option 2: Modify model.py to support delay matrices (future work)
        # ======================================================================
        
        # Calculate mean delay from the delay matrix (excluding self and zeros)
        non_self_delays = self.delay_matrix[~np.eye(self.delay_matrix.shape[0], dtype=bool)]
        connected_delays = non_self_delays[non_self_delays > 0]
        
        if len(connected_delays) > 0:
            mean_delay = connected_delays.mean()
        else:
            # Fallback to config rho if no connections
            mean_delay = self.config['model_params'].get('rho', 10.0)
        
        print(f"\n⚠ MODEL LIMITATION:")
        print(f"  Current model.py uses a single 'rho' value for all connections.")
        print(f"  Using MEAN delay = {mean_delay:.3f} as approximation.")
        print(f"  To use full delay matrix, modify model.py (see TODO in code)")
        
        # Override rho with mean delay
        if params is None:
            params = {}
        params['rho'] = mean_delay
        
        # Initialize model with modified parameters
        self.initialise_model(params=params)
        
        # Store the full delay matrix for analysis/visualization
        self.model.delay_matrix = self.delay_matrix
        print(f"  Full delay matrix stored in model.delay_matrix for analysis")
    
    def visualize_network_and_delays(self, save_path=None):
        """
        Create visualization of network topology and delay matrix.
        
        Generates a two-panel figure:
            - Left: Network graph with nodes at spatial positions
                    Edge colors/widths represent delay magnitudes
            - Right: Delay matrix heatmap
        
        Parameters
        ----------
        save_path : str, optional
            If provided, save figure to this path instead of displaying
            
        Returns
        -------
        fig, axes : tuple
            Matplotlib figure and axes for further customization
        """
        if self.node_positions is None or self.delay_matrix is None:
            raise ValueError("Must compute delay matrix first!")
        
        print("\nGenerating network visualization...")
        
        fig, axes = visualize_network_with_delays(
            network=self.network,
            positions=self.node_positions,
            delay_matrix=self.delay_matrix,
            title=f"Network with Distance-Based Delays"
        )
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved figure to {save_path}")
        else:
            plt.show()
        
        return fig, axes


# ==============================================================================
# MAIN EXPERIMENT EXECUTION
# ==============================================================================
def run_distance_delay_experiment(config_path='experiments/distance_based_delays.yaml'):
    """
    Run a complete distance-based delay experiment.
    
    This is the main entry point for running experiments with distance-based
    delays. It orchestrates the entire workflow:
        1. Load configuration
        2. Create network with spatial positions
        3. Compute distance-based delay matrix
        4. Initialize Wilson-Cowan model
        5. Run simulation
        6. Visualize results
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
        
    Returns
    -------
    model : DistanceDelayedMetapopulation
        The model object with all results
        
    Notes
    -----
    To customize the experiment:
        - Edit distance_based_delays.yaml config file
        - Or pass different config_path
        - Or modify this function to add parameter sweeps
    """
    
    print("=" * 70)
    print("DISTANCE-BASED DELAYS EXPERIMENT")
    print("=" * 70)
    
    # --------------------------------------------------------------------------
    # Step 1: Initialize model and load configuration
    # --------------------------------------------------------------------------
    print("\n[Step 1/6] Initializing model...")
    model = DistanceDelayedMetapopulation()
    model.load_config(config_path)
    
    # Print key configuration parameters
    print(f"\nConfiguration loaded from: {config_path}")
    print(f"  Network topology: {model.config['network_params']['topology']}")
    print(f"  Number of nodes: {model.config['network_params']['N']}")
    print(f"  Distance delays enabled: "
          f"{model.config.get('distance_delay_params', {}).get('enabled', True)}")
    
    # --------------------------------------------------------------------------
    # Step 2: Create network with spatial positions
    # --------------------------------------------------------------------------
    print("\n[Step 2/6] Creating network with spatial positions...")
    model.create_network_with_positions()
    
    # --------------------------------------------------------------------------
    # Step 3: Compute distance-based delay matrix
    # --------------------------------------------------------------------------
    print("\n[Step 3/6] Computing distance-based delays...")
    model.compute_delay_matrix()
    
    # --------------------------------------------------------------------------
    # Step 4: Visualize network and delays BEFORE simulation
    # --------------------------------------------------------------------------
    print("\n[Step 4/6] Visualizing network structure and delays...")
    model.visualize_network_and_delays()
    
    # --------------------------------------------------------------------------
    # Step 5: Initialize Wilson-Cowan model with delays
    # --------------------------------------------------------------------------
    print("\n[Step 5/6] Initializing Wilson-Cowan model...")
    model.initialise_model_with_delays()
    
    # --------------------------------------------------------------------------
    # Step 6: Run simulation
    # --------------------------------------------------------------------------
    print("\n[Step 6/6] Running simulation...")
    duration = model.config['simulation']['duration']
    dt = model.config['simulation']['dt']
    print(f"  Duration: {duration} time units")
    print(f"  Timestep: {dt}")
    
    model.run_simulation(timeit=True)
    
    # --------------------------------------------------------------------------
    # Step 7: Visualize results
    # --------------------------------------------------------------------------
    print("\n[Step 7/6] Visualizing trajectories...")
    model.plot_trajectories()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    return model


# ==============================================================================
# EXAMPLE: Parameter sweep over signal velocity
# ==============================================================================
def parameter_sweep_velocity(velocities=[0.5, 1.0, 2.0, 5.0, 10.0],
                            config_path='experiments/distance_based_delays.yaml'):
    """
    Run multiple experiments with different signal velocities.
    
    This demonstrates how to do parameter sweeps to study the effect
    of delay magnitude on network dynamics.
    
    Parameters
    ----------
    velocities : list of float
        Signal velocity values to test
    config_path : str
        Base configuration file
        
    Notes
    -----
    Increasing velocity → shorter delays → faster communication
    Decreasing velocity → longer delays → slower communication
    
    You can modify this function to sweep other parameters:
        - Coupling strength (k)
        - Network topology
        - Network size (N)
        - Spatial scale
    """
    
    print("=" * 70)
    print("PARAMETER SWEEP: Signal Velocity")
    print("=" * 70)
    
    results = {}
    
    for velocity in velocities:
        print(f"\n\n{'='*70}")
        print(f"Running with signal_velocity = {velocity}")
        print(f"{'='*70}")
        
        # Create model
        model = DistanceDelayedMetapopulation()
        model.load_config(config_path)
        
        # Override signal velocity
        model.config['distance_delay_params']['signal_velocity'] = velocity
        
        # Run experiment
        model.create_network_with_positions()
        model.compute_delay_matrix()
        model.initialise_model_with_delays()
        model.run_simulation(timeit=False)
        
        # Store results
        results[velocity] = {
            'model': model,
            'trajectories': model.model.trajectories,
            'delay_matrix': model.delay_matrix
        }
    
    # Compare results across velocities
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP COMPLETE")
    print("=" * 70)
    
    # Plot comparison
    fig, axes = plt.subplots(len(velocities), 1, figsize=(12, 3*len(velocities)))
    if len(velocities) == 1:
        axes = [axes]
    
    for idx, velocity in enumerate(velocities):
        traj = results[velocity]['trajectories']
        t = results[velocity]['model'].model.time_array
        
        # Plot excitatory populations
        axes[idx].plot(t, traj[0].T, color='blue', alpha=0.5, linewidth=1)
        axes[idx].set_title(f"Signal Velocity = {velocity}")
        axes[idx].set_ylabel("Excitatory Activity")
        if idx == len(velocities) - 1:
            axes[idx].set_xlabel("Time")
    
    plt.tight_layout()
    plt.show()
    
    return results


# ==============================================================================
# RUN SCRIPT
# ==============================================================================
if __name__ == "__main__":
    """
    Main execution block.
    
    Uncomment one of the following to run:
        1. Single experiment with config file settings
        2. Parameter sweep over signal velocities
    """
    
    # Option 1: Single experiment
    # Uses all parameters from distance_based_delays.yaml
    model = run_distance_delay_experiment(
        config_path='experiments/distance_based_delays.yaml'
    )
    
    # Option 2: Parameter sweep (uncomment to run)
    # results = parameter_sweep_velocity(
    #     velocities=[0.5, 1.0, 2.0, 5.0],
    #     config_path='experiments/distance_based_delays.yaml'
    # )
    
    print("\nTo modify experiment parameters:")
    print("  1. Edit experiments/distance_based_delays.yaml")
    print("  2. Or create a new config file and pass path to this script")
    print("  3. Or modify parameters directly in this script")
    print("\nKey parameters to explore:")
    print("  - signal_velocity: Controls delay magnitude")
    print("  - k: Inter-node coupling strength")
    print("  - topology: Network structure")
    print("  - N: Number of nodes")
