"""
Next Generation Neural Mass Model -- Network (Metapopulation) version.

Extends the single-node model (next_gen_model.py) to a network of N coupled
E-I nodes on an arbitrary graph topology, with conduction delays.

Based on:
  - Forrester et al. (2024), Eqs 7-10, 16

Inter-node coupling:
  - Chemical synapses only: E->E across nodes (Eq 10)
    Each directed edge (j -> i) adds 2 ODEs: g_ij, s_ij
    driven by R_{E_j}(t - T_ij)
  - No inter-node gap junctions (biologically, gap junctions only exist
    between nearby neurons within the same cortical region)

State vector layout (per node i = 0 .. N-1):
  [R_E, R_I, V_E, V_I, g_EE, g_EI, g_IE, g_II, s_EE, s_EI, s_IE, s_II]
  = 12 variables per node, indices 12*i .. 12*i + 11

  Then for inter-node connections: for each directed edge (j->i), 2 variables:
  [g_ij, s_ij]  appended after all node variables.

Total dimension: 12*N + 2*M   where M = number of directed edges (j, i) with w_ji > 0.
"""

import numpy as np
from jitcdde import jitcdde, y, t
from symengine import exp, pi, sqrt # faster than sympy
from itertools import product
import networkx as nx


def default_params():
    """Return default Next Gen network parameters (Forrester et al. 2024, p.7)."""
    return dict(
        # --- Local (intra-node) parameters ---
        # Mean population inputs
        eta_I=3.0,
        eta_E=-2.5,

        # Input distributions' widths at half maximum
        delta_I=0.5,
        delta_E=0.5,

        # Membrane timescales
        tau_I=0.012,
        tau_E=0.011,

        # Synaptic rates
        alpha_EE=50.0,
        alpha_EI=40.0,
        alpha_IE=50.0,
        alpha_II=40.0,

        # Synaptic coupling strengths
        kappa_s_EE=0.5,
        kappa_s_EI=0.3,
        kappa_s_IE=0.7,
        kappa_s_II=0.3,

        # Synaptic Reversal potentials
        v_syn_EE=10.0,
        v_syn_EI=-10.0,
        v_syn_IE=10.0,
        v_syn_II=-10.0,

        # Gap junction strengths
        kappa_v_EE=0.01,
        kappa_v_EI=0.0,
        kappa_v_IE=0.0,
        kappa_v_II=0.025,

        # --- Inter-node coupling parameters ---
        # Network coupling strength
        k_ext=0.2,
        # synaptic reversal potentials
        v_syn_ij=10.0,
        # synaptic rates
        alpha_ij=40.0,
        
        # Conduction velocity (m/s) for computing delays from distances
        conduction_velocity=12.0,
        
        # Fixed delay (seconds) 
        delay=0.010,
        # Delay mode: 'constant', 'distance', 'gamma', 'heterogeneous_velocity'
        delay_mode='constant',
        # For gamma-distributed delays: shape parameter (mean = delay)
        delay_gamma_shape=5.0,
        # For heterogeneous velocity: std of velocity distribution
        velocity_std=2.0,
    )



# Per-node state ordering (12 variables per E-I node)
_RE, _RI, _VE, _VI = 0, 1, 2, 3
_GEE, _GEI, _GIE, _GII = 4, 5, 6, 7
_SEE, _SEI, _SIE, _SII = 8, 9, 10, 11
_NODE_DIM = 12


def _node_idx(node_i, var):
    """Global index of variable `var` at node `node_i`."""
    return _NODE_DIM * node_i + var


# =====================================================================
# Main class
# =====================================================================
class NextGenNetwork:
    """Next Generation Neural Mass Model on a network with delays.

    Uses jitcdde to handle delay differential equations arising from
    finite conduction delays between nodes.
    """

    def __init__(self, network, params=None):
        """
        Parameters
        ----------
        network : Network
            Must have attributes: N (int), A (NxN adjacency matrix).
        params : dict or None
            Model parameters; missing keys fall back to default_params().
        """
        self.network = network
        self.p = default_params()
        if params is not None:
            self.p.update(params)

        self.N = network.N
        self.W = np.array(network.A, dtype=float)  # adjacency / weight matrix

        # Row-normalise weights so that afferent strengths sum to 1 per node
        # (Forrester et al. p.9: "normalised by row so that afferent connection strengths for each node sum to unity")
        if self.p.get('normalize_weights', True):
            row_sums = self.W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # avoid division by zero for isolated nodes
            self.W = self.W / row_sums

        # Build directed edge list: (j, i) where W[i, j] > 0
        self.edges = []
        self.edge_weights = []
        for i in range(self.N):
            for j in range(self.N):
                if i != j and self.W[i, j] > 0:
                    self.edges.append((j, i))
                    self.edge_weights.append(self.W[i, j])
        self.M = len(self.edges) # number of edges

        # Total state dimension
        self.n_vars = _NODE_DIM * self.N + 2 * self.M

        # Build delay matrix (seconds)
        self._build_delay_matrix()

        # Storage
        self.time_array = None
        self.trajectories = None  # shape (n_vars, T)
        self.components = ['R_E', 'R_I', 'V_E', 'V_I']

    # ------------------------------------------------------------------
    def _build_delay_matrix(self):
        """Compute delay T_ij for each edge based on the chosen delay mode.

        Delay modes:
            'constant'    : All edges share the same delay.
            'matrix'      : T_ij = D[i,j] / v, where D is an external distance
                            matrix (e.g. from HCP centroid distances).
                            Requires params['distance_matrix'] = ndarray (N,N) in mm.
                            This is closest to the paper: T_ij = d_ij / v (p.7).
            'gamma'       : Delays drawn from a Gamma distribution.
            'distance'    : T_ij = shortest_path(j,i) / v (graph hops, not real distance).
            'heterogeneous_velocity' : Distance-dependent with per-edge velocity noise.
        """
        mode = self.p.get('delay_mode', 'constant')
        fixed_delay = self.p['delay']
        v = self.p['conduction_velocity']

        if mode == 'constant':
            self.delays = np.full(self.M, fixed_delay)

        elif mode == 'matrix':
            # Use external distance matrix: T_ij = D[i,j] / v
            # This matches the paper: T_ij = d_ij / v, v = 12 m/s (p.7)
            D = self.p.get('distance_matrix')
            if D is None:
                raise ValueError("delay_mode='matrix' requires params['distance_matrix']")
            # D is in mm, v is in m/s → convert: T = D(mm) / (v(m/s) * 1000) = D / (v*1000) seconds
            self.delays = np.array([
                D[i, j] / (v * 1000.0)  # mm / (mm/s) = seconds
                for (j, i) in self.edges
            ])
            # Ensure minimum delay (avoid zero for numerical stability)
            self.delays = np.maximum(self.delays, 1e-4)

        elif mode == 'gamma':
            # Gamma distribution: mean = fixed_delay, shape = k
            k = self.p.get('delay_gamma_shape', 5.0)
            theta = fixed_delay / k  # scale parameter so mean = k*theta
            rng = np.random.default_rng(42)
            self.delays = rng.gamma(k, theta, size=self.M)
            self.delays = np.maximum(self.delays, 1e-4)

        elif mode == 'distance':
            # Compute shortest-path distances on the graph
            G = self.network.network
            sp = dict(nx.shortest_path_length(G))
            self.delays = np.array([
                max(sp[j][i], 1) / v
                for (j, i) in self.edges
            ])

        elif mode == 'heterogeneous_velocity':
            # Distance-dependent with varying conduction velocity per edge
            G = self.network.network
            sp = dict(nx.shortest_path_length(G))
            sigma_v = self.p.get('velocity_std', 2.0)
            rng = np.random.default_rng(42)
            velocities = rng.normal(v, sigma_v, size=self.M)
            velocities = np.maximum(velocities, 1.0)
            self.delays = np.array([
                max(sp[j][i], 1) / vel
                for (j, i), vel in zip(self.edges, velocities)
            ])

        else:
            raise ValueError(f"Unknown delay_mode: {mode}")

        self.delays = np.array(self.delays, dtype=float)

    # ------------------------------------------------------------------
    def _edge_var_idx(self, edge_k, var_offset):
        """Global index of inter-node variable for edge k.

        var_offset: 0 for g_ij, 1 for s_ij
        """
        return _NODE_DIM * self.N + 2 * edge_k + var_offset

    # ------------------------------------------------------------------
    # Build the jitcdde system
    # ------------------------------------------------------------------
    def build_dde(self):
        """Construct the jitcdde DDE system.

        Returns
        -------
        DDE : jitcdde instance
        """
        p = self.p
        N = self.N

        # Precompute: for each node i, list of (edge_k, j, w_ij, T_ij)
        incoming = {i: [] for i in range(N)}
        for k, (j, i) in enumerate(self.edges):
            incoming[i].append((k, j, self.edge_weights[k], self.delays[k]))

        f = []  # will hold n_vars symbolic expressions

        # ---- Extract parameters ----
        eta_E, eta_I = p['eta_E'], p['eta_I']
        delta_E, delta_I = p['delta_E'], p['delta_I']
        tau_E, tau_I = p['tau_E'], p['tau_I']
        alpha_EE, alpha_EI, alpha_IE, alpha_II = p['alpha_EE'], p['alpha_EI'], p['alpha_IE'], p['alpha_II']
        kappa_v_EE, kappa_v_EI, kappa_v_IE, kappa_v_II = p['kappa_v_EE'], p['kappa_v_EI'], p['kappa_v_IE'], p['kappa_v_II']
        v_syn_EE, v_syn_EI, v_syn_IE, v_syn_II = p['v_syn_EE'], p['v_syn_EI'], p['v_syn_IE'], p['v_syn_II']
        kappa_s_EE, kappa_s_EI, kappa_s_IE, kappa_s_II = p['kappa_s_EE'], p['kappa_s_EI'], p['kappa_s_IE'], p['kappa_s_II']
        # Inter-node parameters
        v_syn_ij = p['v_syn_ij']
        alpha_ij = p['alpha_ij']
        k_ext = p['k_ext']

        # ---- Node equations (12 per node) ----
        for node in range(N):
            # state: map local variable offsets to global indices
            R_E  = _node_idx(node, _RE)
            R_I  = _node_idx(node, _RI)
            V_E  = _node_idx(node, _VE)
            V_I  = _node_idx(node, _VI)
            g_EE = _node_idx(node, _GEE)
            g_EI = _node_idx(node, _GEI)
            g_IE = _node_idx(node, _GIE)
            g_II = _node_idx(node, _GII)
            s_EE = _node_idx(node, _SEE)
            s_EI = _node_idx(node, _SEI)
            s_IE = _node_idx(node, _SIE)
            s_II = _node_idx(node, _SII)

            # --- Inter-node conductances arriving at this node's E population ---
            sum_g_ext = 0
            sum_gsyn_ext = 0
            for (k, j, w_ij, T_ij) in incoming[node]:
                g_ij_idx = self._edge_var_idx(k, 0)
                sum_g_ext = sum_g_ext + y(g_ij_idx)
                sum_gsyn_ext = sum_gsyn_ext + y(g_ij_idx) * (v_syn_ij - y(V_E))

            # --- Firing-rate equations (Eq 7) ---
            # Sum of conductances + gap-junction terms: sum_b (g_ab + kappa_v_ab)
            sum_g_kv_E = (y(g_EE) + kappa_v_EE) + (y(g_EI) + kappa_v_EI)
            sum_g_kv_I = (y(g_IE) + kappa_v_IE) + (y(g_II) + kappa_v_II)

            # E population also receives inter-node conductances
            sum_g_kv_E_total = sum_g_kv_E + sum_g_ext

            dR_E = (-y(R_E) * sum_g_kv_E_total + 2.0 * y(R_E) * y(V_E) + delta_E / (np.pi * tau_E)) / tau_E
            f.append(dR_E)

            # I population: no inter-node coupling
            dR_I = (-y(R_I) * sum_g_kv_I + 2.0 * y(R_I) * y(V_I) + delta_I / (np.pi * tau_I)) / tau_I
            f.append(dR_I)

            # --- Mean-voltage equations (Eq 8) ---
            # Synaptic current: sum_b g_ab * (v_syn_ab - V_a)
            I_syn_E = (y(g_EE) * (v_syn_EE - y(V_E)) + y(g_EI) * (v_syn_EI - y(V_E)))
            I_syn_I = (y(g_IE) * (v_syn_IE - y(V_I)) + y(g_II) * (v_syn_II - y(V_I)))

            # Gap-junction current: sum_b kappa_v_ab * (V_b - V_a)
            I_gap_E = (kappa_v_EE * (y(V_E) - y(V_E)) + kappa_v_EI * (y(V_I) - y(V_E)))
            I_gap_I = (kappa_v_IE * (y(V_E) - y(V_I)) + kappa_v_II * (y(V_I) - y(V_I)))

            dV_E = (eta_E + y(V_E)**2 - (np.pi * tau_E)**2 * y(R_E)**2
                    + I_syn_E + I_gap_E
                    + sum_gsyn_ext   # inter-node chemical (Eq 10)
                    ) / tau_E
            f.append(dV_E)

            dV_I = (eta_I + y(V_I)**2 - (np.pi * tau_I)**2 * y(R_I)**2
                    + I_syn_I + I_gap_I
                    ) / tau_I
            f.append(dV_I)

            # --- Synaptic conductance equations (Eq 9) ---
            f.append(alpha_EE * (y(s_EE) - y(g_EE)))   # dg_EE
            f.append(alpha_EI * (y(s_EI) - y(g_EI)))   # dg_EI
            f.append(alpha_IE * (y(s_IE) - y(g_IE)))   # dg_IE
            f.append(alpha_II * (y(s_II) - y(g_II)))   # dg_II

            # --- Synaptic gating equations (Eq 9) ---
            f.append(alpha_EE * (kappa_s_EE * y(R_E) - y(s_EE)))  # ds_EE
            f.append(alpha_EI * (kappa_s_EI * y(R_I) - y(s_EI)))  # ds_EI
            f.append(alpha_IE * (kappa_s_IE * y(R_E) - y(s_IE)))  # ds_IE
            f.append(alpha_II * (kappa_s_II * y(R_I) - y(s_II)))  # ds_II

        # ---- Inter-node edge equations (2 per directed edge) ----
        for k, (j, i) in enumerate(self.edges):
            g_ij_idx = self._edge_var_idx(k, 0)
            s_ij_idx = self._edge_var_idx(k, 1)
            T_ij = self.delays[k]
            w_ij = self.edge_weights[k]

            RE_j = _node_idx(j, _RE)   # source node's R_E

            # dg_ij / dt = alpha_ij * (s_ij - g_ij)       [Eq 9a analog]
            f.append(alpha_ij * (y(s_ij_idx) - y(g_ij_idx)))

            # ds_ij / dt = alpha_ij * (k_ext * w_ij * R_{E_j}(t - T_ij) - s_ij)
            if T_ij > 0:
                source_RE_delayed = y(RE_j, t - T_ij)
            else:
                source_RE_delayed = y(RE_j)

            f.append(alpha_ij * (k_ext * w_ij * source_RE_delayed - y(s_ij_idx)))

        assert len(f) == self.n_vars, \
            f"Expected {self.n_vars} equations, got {len(f)}"

        DDE = jitcdde(f)
        return DDE

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def run(self, duration=2.0, dt=0.001, y0=None):
        """Integrate the network model.

        Parameters
        ----------
        duration : float
            Simulation length in seconds.
        dt : float
            Output time-step (seconds).
        y0 : array-like or None
            Initial conditions (length n_vars). If None, use defaults.

        Returns
        -------
        time_array : ndarray, shape (T,)
        trajectories : ndarray, shape (n_vars, T)
        """
        # Build DDE system
        DDE = self.build_dde()

        # Initial conditions
        if y0 is None:
            y0 = np.zeros(self.n_vars)
            for node in range(self.N):
                base = _NODE_DIM * node
                y0[base + _RE] = 0.1
                y0[base + _RI] = 0.1
                y0[base + _VE] = -2.0
                y0[base + _VI] = -2.0
                # g and s variables start at small positive values
                for offset in range(_GEE, _SII + 1):
                    y0[base + offset] = 0.01
            # Inter-node g_ij, s_ij start at 0
            # (already zero from np.zeros)

        DDE.constant_past(y0)
        DDE.adjust_diff()

        # Time grid
        time_array = np.arange(0, duration, dt)

        # Integrate
        results = np.empty((len(time_array), self.n_vars))
        for idx, t_val in enumerate(time_array):
            if t_val <= DDE.t:
                results[idx, :] = DDE.integrate(DDE.t)
            else:
                results[idx, :] = DDE.integrate(t_val)

        self.time_array = time_array
        self.trajectories = results.T  # shape (n_vars, T)
        return self.time_array, self.trajectories

    # ------------------------------------------------------------------
    # Extract node-level variables
    # ------------------------------------------------------------------
    def get_node_var(self, node_i, var_name):
        """Get a time series for a specific variable at a node.

        Parameters
        ----------
        node_i : int
        var_name : str
            One of 'R_E', 'R_I', 'V_E', 'V_I', 'g_EE', etc.

        Returns
        -------
        ts : ndarray, shape (T,)
        """
        var_map = {
            'R_E': _RE, 'R_I': _RI, 'V_E': _VE, 'V_I': _VI,
            'g_EE': _GEE, 'g_EI': _GEI, 'g_IE': _GIE, 'g_II': _GII,
            's_EE': _SEE, 's_EI': _SEI, 's_IE': _SIE, 's_II': _SII,
        }
        idx = _node_idx(node_i, var_map[var_name])
        return self.trajectories[idx]

    def get_all_RE(self):
        """Return R_E for all nodes, shape (N, T)."""
        return np.array([self.get_node_var(i, 'R_E') for i in range(self.N)])

    def get_all_VE(self):
        """Return V_E for all nodes, shape (N, T)."""
        return np.array([self.get_node_var(i, 'V_E') for i in range(self.N)])

    # ------------------------------------------------------------------
    # Kuramoto order parameter Z (per node)
    # ------------------------------------------------------------------
    def compute_Z(self):
        """Compute Kuramoto order parameter Z for each node's E and I populations.

        Returns
        -------
        Z_E : ndarray, shape (N, T) -- complex
        Z_I : ndarray, shape (N, T) -- complex
        """
        if self.trajectories is None:
            raise RuntimeError("Call run() first.")

        p = self.p
        Z_E = np.empty((self.N, len(self.time_array)), dtype=complex)
        Z_I = np.empty_like(Z_E)

        for i in range(self.N):
            R_E = self.get_node_var(i, 'R_E')
            R_I = self.get_node_var(i, 'R_I')
            V_E = self.get_node_var(i, 'V_E')
            V_I = self.get_node_var(i, 'V_I')

            W_E = np.pi * p['tau_E'] * R_E + 1j * V_E
            W_I = np.pi * p['tau_I'] * R_I + 1j * V_I

            Z_E[i] = (1.0 - np.conj(W_E)) / (1.0 + np.conj(W_E))
            Z_I[i] = (1.0 - np.conj(W_I)) / (1.0 + np.conj(W_I))

        return Z_E, Z_I

    # ------------------------------------------------------------------
    # Phase Locking Value (PLV) -- functional connectivity
    # ------------------------------------------------------------------
    def compute_PLV(self, population='E', t_start=None):
        """Compute PLV matrix from Kuramoto order parameter phases.

        PLV_ij = |mean(exp(i * (theta_i(t) - theta_j(t))))|

        Parameters
        ----------
        population : str
            'E' or 'I' -- which population's Z to use.
        t_start : float or None
            Discard transient before this time (seconds).
            Default: discard first 25% of simulation.

        Returns
        -------
        PLV : ndarray, shape (N, N)
        """
        Z_E, Z_I = self.compute_Z()
        Z = Z_E if population == 'E' else Z_I

        # Discard transient
        if t_start is None:
            t_start = self.time_array[-1] * 0.25
        mask = self.time_array >= t_start
        Z_trimmed = Z[:, mask]

        # Extract phases
        theta = np.angle(Z_trimmed)  # shape (N, T')

        # Compute PLV matrix
        N = self.N
        PLV = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                phase_diff = theta[i] - theta[j]
                plv_ij = np.abs(np.mean(np.exp(1j * phase_diff)))
                PLV[i, j] = plv_ij
                PLV[j, i] = plv_ij
            PLV[i, i] = 1.0

        return PLV

    # ------------------------------------------------------------------
    # Pearson correlation FC (using R_E time series)
    # ------------------------------------------------------------------
    def compute_FC_corr(self, t_start=None):
        """Compute Pearson correlation FC matrix from R_E time series.

        Parameters
        ----------
        t_start : float or None
            Discard transient before this time.

        Returns
        -------
        FC : ndarray, shape (N, N)
        """
        if t_start is None:
            t_start = self.time_array[-1] * 0.25
        mask = self.time_array >= t_start

        RE_all = self.get_all_RE()[:, mask]  # (N, T')
        FC = np.corrcoef(RE_all)
        return FC

