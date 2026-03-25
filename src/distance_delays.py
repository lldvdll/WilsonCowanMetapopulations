"""
Distance-Based Delay Module for Metapopulation Wilson-Cowan Model

This module provides functions to:
1. Assign spatial coordinates to network nodes based on topology
2. Calculate Euclidean distances between nodes
3. Generate distance-based inter-node delay matrices

Usage:
    from src.distance_delays import assign_node_coordinates, compute_delay_matrix
    
    # Get node positions based on network topology
    positions = assign_node_coordinates(network, topology='ring')
    
    # Compute delay matrix from positions
    delay_matrix = compute_delay_matrix(
        positions, 
        signal_velocity=2.0,
        base_delay=0.5
    )

Key Concepts:
    - Inter-node delay ρ_nj = base_delay + (distance_nj / signal_velocity)
    - Distance is Euclidean: sqrt((x_n - x_j)² + (y_n - y_j)²)
    - Only applies to CONNECTED nodes (based on adjacency matrix)
    - Self-connections (n=j) can have a separate self_delay value

Author: Your Name
Date: 2025
"""

import numpy as np
import networkx as nx


def assign_node_coordinates(network, topology=None, scale=1.0, **kwargs):
    """
    Assign (x, y) spatial coordinates to each node in the network based on topology.
    
    This function creates a spatial embedding of the network, which is used to
    calculate inter-node distances for distance-based delays.
    
    Parameters
    ----------
    network : Network object or networkx.Graph
        The network object containing the graph structure
    topology : str, optional
        Network topology type. If None, will try to infer from network.config
        Options: 'line', 'ring', 'lattice', 'full', 'smallworld'
    scale : float, optional (default=1.0)
        Scaling factor for coordinates. Increase to spread nodes further apart.
        Larger scale = larger distances = larger delays (for fixed signal velocity)
    **kwargs : dict
        Additional parameters for specific topologies
        
    Returns
    -------
    positions : dict
        Dictionary mapping node index (int) to (x, y) tuple of coordinates
        Example: {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), ...}
        
    Notes
    -----
    - Line topology: Nodes placed along x-axis, evenly spaced
    - Ring topology: Nodes placed on a circle
    - Lattice topology: Nodes placed on a 2D grid
    - Full/SmallWorld: Spring layout (force-directed) for aesthetics
    
    Examples
    --------
    >>> network = Network({'N': 10, 'topology': 'ring'})
    >>> positions = assign_node_coordinates(network, scale=2.0)
    >>> print(positions[0])  # First node position
    (2.0, 0.0)
    
    To change the spatial scale (affects all distances):
    >>> positions = assign_node_coordinates(network, scale=5.0)  # Nodes 5x further apart
    """
    
    # Extract the networkx graph from Network object
    if hasattr(network, 'network'):
        G = network.network
        N = network.N
        if topology is None and hasattr(network, 'config'):
            topology = network.config.get('topology', 'full')
    else:
        G = network
        N = len(G.nodes())
    
    positions = {}
    
    # ==========================================
    # LINE TOPOLOGY: Nodes along x-axis
    # ==========================================
    if topology == 'line':
        # Place nodes at (0, 0), (1, 0), (2, 0), ..., (N-1, 0)
        for i in range(N):
            positions[i] = (i * scale, 0.0)
    
    # ==========================================
    # RING TOPOLOGY: Nodes on a circle
    # ==========================================
    elif topology == 'ring':
        # Place nodes evenly around a circle of radius = N/(2π)
        # This ensures perimeter ≈ N, so adjacent nodes are ~1 unit apart (before scaling)
        radius = N / (2 * np.pi)
        for i in range(N):
            angle = 2 * np.pi * i / N  # Evenly space around circle
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions[i] = (x * scale, y * scale)
    
    # ==========================================
    # LATTICE TOPOLOGY: 2D square grid
    # ==========================================
    elif topology == 'lattice':
        # Arrange nodes in a square grid
        # If N=16, make a 4x4 grid. If N=20, make a 5x4 grid, etc.
        side = int(np.ceil(np.sqrt(N)))  # Grid dimension
        for i in range(N):
            row = i // side  # Which row (y-coordinate)
            col = i % side   # Which column (x-coordinate)
            positions[i] = (col * scale, row * scale)
    
    # ==========================================
    # FULL or SMALLWORLD: Spring layout
    # ==========================================
    elif topology in ['full', 'smallworld']:
        # Use NetworkX's spring layout (force-directed algorithm)
        # This spreads nodes out aesthetically, but positions are somewhat arbitrary
        # Seed for reproducibility
        seed = kwargs.get('seed', 42)
        pos_dict = nx.spring_layout(G, scale=scale, seed=seed)
        positions = {i: (pos_dict[i][0], pos_dict[i][1]) for i in range(N)}
    
    else:
        raise ValueError(f"Unknown topology: {topology}. Use 'line', 'ring', 'lattice', 'full', or 'smallworld'")
    
    return positions


def compute_distance_matrix(positions):
    """
    Compute Euclidean distance matrix from node positions.
    
    Parameters
    ----------
    positions : dict
        Dictionary mapping node index to (x, y) coordinates
        Example: {0: (0, 0), 1: (1, 0), 2: (2, 0)}
        
    Returns
    -------
    distance_matrix : np.ndarray, shape (N, N)
        Matrix where entry [i, j] = Euclidean distance from node i to node j
        Diagonal entries (i=i) are zero
        
    Notes
    -----
    Distance formula: d_ij = sqrt((x_i - x_j)² + (y_i - y_j)²)
    
    Examples
    --------
    >>> positions = {0: (0, 0), 1: (1, 0), 2: (0, 1)}
    >>> D = compute_distance_matrix(positions)
    >>> print(D[0, 1])  # Distance from node 0 to node 1
    1.0
    >>> print(D[0, 2])  # Distance from node 0 to node 2
    1.0
    >>> print(D[1, 2])  # Distance from node 1 to node 2
    1.414...  # sqrt(2)
    """
    N = len(positions)
    distance_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                x_i, y_i = positions[i]
                x_j, y_j = positions[j]
                distance_matrix[i, j] = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
    
    return distance_matrix


def compute_delay_matrix(positions, adjacency_matrix=None, signal_velocity=1.0, 
                         base_delay=0.0, self_delay=0.0, max_delay=None):
    """
    Compute inter-node delay matrix from spatial positions.
    
    This is the main function for generating distance-based delays.
    
    Parameters
    ----------
    positions : dict
        Node positions as {node_id: (x, y)}
    adjacency_matrix : np.ndarray, optional
        Network adjacency matrix (N x N). If provided, delays are only computed
        for connected nodes (where A[i,j] != 0). Unconnected nodes get delay = 0.
    signal_velocity : float, optional (default=1.0)
        Speed of signal propagation between nodes. 
        Higher velocity = shorter delays for same distance.
        Units: [distance units / time units]
        Example: If distance in mm and time in ms, velocity in mm/ms
        
        **TO CHANGE DELAYS: Adjust this value**
        - Increase signal_velocity → shorter delays (faster transmission)
        - Decrease signal_velocity → longer delays (slower transmission)
        
    base_delay : float, optional (default=0.0)
        Minimum delay added to all connections, regardless of distance.
        Represents synaptic/processing delay independent of transmission time.
        Units: [time units]
        
        **TO CHANGE MINIMUM DELAY: Adjust this value**
        
    self_delay : float, optional (default=0.0)
        Delay for self-connections (node i to itself).
        Usually set to 0 or a small value.
        
    max_delay : float, optional (default=None)
        Maximum allowed delay. If specified, any computed delay exceeding this
        will be capped at max_delay. Useful for computational efficiency.
        
        **TO LIMIT LARGE DELAYS: Set this value**
        Example: max_delay=50.0 ensures no delay exceeds 50 time units
        
    Returns
    -------
    delay_matrix : np.ndarray, shape (N, N)
        Matrix where entry [i, j] = delay for signal from node i to node j
        
        Formula: delay[i, j] = base_delay + (distance[i, j] / signal_velocity)
        
        Special cases:
        - If i == j: delay = self_delay
        - If A[i, j] == 0 (not connected): delay = 0
        
    Notes
    -----
    The delay matrix corresponds to ρ_nj in the paper's equations.
    
    Physical interpretation:
        If a signal travels distance d at velocity v, it takes time t = d/v
        
    Examples
    --------
    >>> positions = {0: (0, 0), 1: (3, 4)}  # Distance = 5
    >>> delays = compute_delay_matrix(positions, signal_velocity=2.0, base_delay=1.0)
    >>> print(delays[0, 1])  # base_delay + distance/velocity = 1.0 + 5/2 = 3.5
    3.5
    
    To make delays shorter (faster signal):
    >>> delays = compute_delay_matrix(positions, signal_velocity=10.0)  # 10x faster
    
    To add uniform processing delay:
    >>> delays = compute_delay_matrix(positions, base_delay=2.0)  # All delays +2
    
    To cap maximum delays:
    >>> delays = compute_delay_matrix(positions, max_delay=20.0)  # No delay > 20
    """
    
    N = len(positions)
    
    # First, compute the distance matrix
    distance_matrix = compute_distance_matrix(positions)
    
    # Convert distances to delays using the formula:
    # delay = base_delay + (distance / signal_velocity)
    delay_matrix = base_delay + (distance_matrix / signal_velocity)
    
    # Set self-delays (diagonal entries)
    for i in range(N):
        delay_matrix[i, i] = self_delay
    
    # Apply adjacency matrix mask if provided
    # Only connected nodes should have non-zero delays
    if adjacency_matrix is not None:
        # Set delay to 0 for unconnected node pairs
        delay_matrix = delay_matrix * (adjacency_matrix > 0)
        # But preserve self-delays even if A[i,i] = 0
        for i in range(N):
            delay_matrix[i, i] = self_delay
    
    # Apply maximum delay cap if specified
    if max_delay is not None:
        delay_matrix = np.minimum(delay_matrix, max_delay)
    
    return delay_matrix


def visualize_network_with_delays(network, positions, delay_matrix, 
                                   title="Network with Distance-Based Delays"):
    """
    Visualize the network with node positions and edge delays.
    
    Parameters
    ----------
    network : Network object
        Network object containing the graph
    positions : dict
        Node positions {node_id: (x, y)}
    delay_matrix : np.ndarray
        Inter-node delay matrix (N x N)
    title : str
        Plot title
        
    Returns
    -------
    fig, (ax1, ax2) : tuple
        Figure and axes objects for further customization
        
    Notes
    -----
    - Left subplot: Network graph with nodes at spatial positions
    - Right subplot: Delay matrix as a heatmap
    
    Edge colors/widths represent delay magnitudes:
        - Thicker edges = longer delays
        - Color scale: blue (short delay) to red (long delay)
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    G = network.network if hasattr(network, 'network') else network
    N = len(positions)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ==========================================
    # LEFT PLOT: Network graph with positions
    # ==========================================
    
    # Get delays for all edges (for coloring)
    edges = list(G.edges())
    if len(edges) > 0:
        edge_delays = [delay_matrix[i, j] for i, j in edges]
        max_edge_delay = max(edge_delays) if edge_delays else 1.0
        min_edge_delay = min(edge_delays) if edge_delays else 0.0
        
        # Normalize delays to [0, 1] for colormap
        if max_edge_delay > min_edge_delay:
            edge_colors = [(d - min_edge_delay) / (max_edge_delay - min_edge_delay) 
                          for d in edge_delays]
        else:
            edge_colors = [0.5] * len(edges)
        
        # Draw network
        nx.draw_networkx_nodes(G, positions, node_color='lightblue', 
                              node_size=300, ax=ax1)
        
        # Draw edges with colors representing delays
        cmap = cm.get_cmap('coolwarm')  # Blue (cold/short) to Red (hot/long)
        edge_collection = nx.draw_networkx_edges(
            G, positions, 
            edge_color=edge_colors,
            edge_cmap=cmap,
            width=2,
            ax=ax1
        )
        
        # Add colorbar for edge delays
        sm = cm.ScalarMappable(cmap=cmap, 
                              norm=plt.Normalize(vmin=min_edge_delay, 
                                                vmax=max_edge_delay))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, label='Inter-node Delay')
        
        # Draw node labels
        nx.draw_networkx_labels(G, positions, font_size=10, ax=ax1)
    
    ax1.set_title(f"Network Topology: {network.config.get('topology', 'Unknown')}")
    ax1.axis('off')
    
    # ==========================================
    # RIGHT PLOT: Delay matrix heatmap
    # ==========================================
    im = ax2.imshow(delay_matrix, cmap='viridis', aspect='auto')
    ax2.set_title("Delay Matrix (ρ_nj)")
    ax2.set_xlabel("Target Node j")
    ax2.set_ylabel("Source Node i")
    plt.colorbar(im, ax=ax2, label='Delay (time units)')
    
    # Add text annotations for small networks (N <= 10)
    if N <= 10:
        for i in range(N):
            for j in range(N):
                text = ax2.text(j, i, f'{delay_matrix[i, j]:.1f}',
                              ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    return fig, (ax1, ax2)


# ==========================================
# USAGE EXAMPLE (if this file is run directly)
# ==========================================
if __name__ == "__main__":
    print("Distance-Based Delays Module")
    print("=" * 50)
    print("\nThis module provides functions for:")
    print("  1. Assigning spatial coordinates to network nodes")
    print("  2. Computing Euclidean distance matrices")
    print("  3. Generating distance-based delay matrices")
    print("\nImport this module in your experiments:")
    print("  from src.distance_delays import compute_delay_matrix")
    print("\nSee experiments/distance_based_delays.py for usage examples")
