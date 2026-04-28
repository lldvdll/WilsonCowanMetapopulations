"""
Delay matrix generators for the metapopulation Wilson-Cowan model.

Three regimes:
    1. uniform_delay_matrix   - scalar rho broadcast to all pairs (paper baseline)
    2. distance_delay_matrix  - rho_nj = d_nj / v, single conduction velocity
    3. heterogeneous_delay_matrix - rho_nj = d_nj / v_nj, v_nj ~ Gamma(p, q)
       per-edge velocities, symmetric matrix

Biological grounding for Gamma parameters:
    Atay & Hutt (2006) show cortico-cortical axonal speeds follow a Gamma
    distribution with mode between 5-12 m/s (rat data, their Figure 1, eq 5.2).
    vm: mode of distribution (m/s)
    p:  shape parameter, controls heterogeneity (low p = high variance)
    q:  scale, set via q = vm / (p - 1)
    vl, vh: physiological bounds (default 1, 20 m/s)
"""

import numpy as np
import networkx as nx


def uniform_delay_matrix(network, rho=10.0):
    """
    Uniform inter-node delay, replicates Conti & Van Gorder (2019) baseline.

    Parameters
    ----------
    network : Network
    rho     : float, delay applied to all node pairs

    Returns
    -------
    D : (N, N) array, D[n,j] = rho for n != j, 0 on diagonal
    """
    N = network.N
    D = np.full((N, N), rho)
    np.fill_diagonal(D, 0.0)
    return D


def distance_delay_matrix(network, v=6.0, target_mean_rho=None):
    """
    Distance-based inter-node delay: rho_nj = d_nj / v
    Distance is graph shortest-path hop count.
    Optionally rescaled so mean delay matches target_mean_rho.

    Parameters
    ----------
    network         : Network
    v               : float, conduction velocity
    target_mean_rho : float or None, if set rescales matrix so mean delay
                      equals this value, enabling fair comparison to uniform baseline

    Returns
    -------
    D : (N, N) array
    """
    assert nx.is_connected(network.network), "Network must be connected"

    N = network.N
    D = np.zeros((N, N))

    lengths = dict(nx.shortest_path_length(network.network))
    for n in range(N):
        for j in range(N):
            if n != j:
                D[n, j] = lengths[n][j] / v

    if target_mean_rho is not None:
        mask = ~np.eye(N, dtype=bool)
        current_mean = np.mean(D[mask])
        D = D * (target_mean_rho / current_mean)

    return D


def heterogeneous_delay_matrix(network, vm=6.0, p=4.5, vl=1.0, vh=20.0, target_mean_rho=None, seed=None):
    """
    Heterogeneous inter-node delay: rho_nj = d_nj / v_nj
    v_nj ~ truncated Gamma(p, q) per edge, symmetric (v_nj == v_jn).

    Follows Atay & Hutt (2006) eq 5.2, motivated by experimental data
    showing cortico-cortical axonal speeds are Gamma-distributed.

    Parameters
    ----------
    network         : Network
    vm              : float, mode of Gamma distribution (m/s), default 6.0
    p               : float, shape parameter (p > 2), controls heterogeneity
                      low p = high variance = more heterogeneous myelination
    vl              : float, lower velocity bound (m/s), default 1.0
    vh              : float, upper velocity bound (m/s), default 20.0
    target_mean_rho : float or None, if set rescales matrix so mean delay
                      equals this value
    seed            : int or None, random seed for reproducibility

    Returns
    -------
    D : (N, N) array
    """
    assert nx.is_connected(network.network), "Network must be connected"
    assert p > 2, "Shape parameter p must be > 2 (required for finite variance)"

    rng = np.random.default_rng(seed)
    N = network.N
    q = vm / (p - 1)

    lengths = dict(nx.shortest_path_length(network.network))
    dist = np.zeros((N, N))
    for n in range(N):
        for j in range(N):
            if n != j:
                dist[n, j] = lengths[n][j]

    # Sample per-edge velocities from truncated Gamma, symmetric
    V = np.zeros((N, N))
    for n in range(N):
        for j in range(n+1, N):
            v_sample = 0.0
            while not (vl <= v_sample <= vh):
                v_sample = rng.gamma(shape=p, scale=q)
            V[n, j] = v_sample
            V[j, n] = v_sample

    D = np.zeros((N, N))
    mask = V > 0
    D[mask] = dist[mask] / V[mask]

    if target_mean_rho is not None:
        off_diag = ~np.eye(N, dtype=bool)
        current_mean = np.mean(D[off_diag])
        D = D * (target_mean_rho / current_mean)

    return D