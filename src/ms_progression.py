"""
MS disease progression utilities.
Simulates progressive demyelination and axonal loss in the network.

Stages:
    0 - Healthy: normal myelination, full connectivity
    1 - Early MS: focal demyelination, increased delay heterogeneity
    2 - Moderate MS: widespread demyelination, high heterogeneity
    3 - Severe MS: axonal loss, reduced connectivity

References:
    Compston & Coles (2008) - MS disease progression
    Waxman (2006) - conduction velocity reduction in demyelination
    Bjartmar & Trapp (2001) - axonal loss in progressive MS
"""

import numpy as np
import networkx as nx


# Stage parameters
MS_STAGES = {
    0: {'vm': 6.0, 'p': 10.0, 'description': 'Healthy baseline'},
    1: {'vm': 4.0, 'p': 4.5,  'description': 'Early MS: focal demyelination'},
    2: {'vm': 2.0, 'p': 2.5,  'description': 'Moderate MS: widespread demyelination'},
    3: {'vm': 2.0, 'p': 2.5,  'description': 'Severe MS: axonal loss'},
}

# Topology-specific edge removal rates for stage 3
REMOVAL_RATES = {
    'line':    0.00,
    'full':    0.30,
    'ring':    0.20,
    'lattice': 0.25,
}


def remove_edges(network, topology, seed=42):
    """
    Remove edges from network to simulate axonal loss in stage 3.
    Updates network.A in place with the damaged adjacency matrix.

    Line topology: systematic removal from ends inward (length-dependent vulnerability)
    All others: random removal at topology-specific rate

    Parameters
    ----------
    network  : Network object
    topology : str
    seed     : int, random seed for reproducibility

    Returns
    -------
    removal_count : int, number of edges removed
    """
    G = network.network.copy()
    rate = REMOVAL_RATES[topology]
    if rate == 0:
        return 0
    edges = list(G.edges())
    n_remove = max(1, int(len(edges) * rate))

    if topology == 'line':
        # Remove from both ends inward, at least 1 edge per end
        n_each_end = max(1, n_remove // 2)
        edges_to_remove = edges[:n_each_end] + edges[-n_each_end:]
        # Deduplicate in case of overlap
        edges_to_remove = list(set(edges_to_remove))
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(edges), size=n_remove, replace=False)
        edges_to_remove = [edges[i] for i in idx]

    G.remove_edges_from(edges_to_remove)

    # Connectivity check
    if not nx.is_connected(G):
        raise ValueError(
            f"Network fragmented after edge removal for topology={topology}. "
            f"Reduce removal rate in REMOVAL_RATES."
        )

    # Update network object in place
    network.network = G
    network.A = nx.to_numpy_array(G)

    return len(edges_to_remove)


def get_stage_params(stage):
    """Return delay parameters for a given MS stage."""
    return MS_STAGES[stage]