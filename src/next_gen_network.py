"""
Next Generation Neural Mass Model -- Network (Metapopulation) version.

Extends the single-node model (next_gen_model.py) to a network of N coupled
E-I nodes on an arbitrary graph topology, with conduction delays.

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
        # When delay_mode='matrix_gamma_velocity', this is the MEAN of the
        # Gamma distribution of per-edge velocities (E[v] = p * q = v_mean).
        conduction_velocity=12.0,

        # Fixed delay (seconds) used by 'constant' mode (and 'gamma' mean)
        delay=0.010,

        # Delay mode (see _build_delay_matrix docstring for full list):
        #   'constant', 'matrix', 'matrix_gamma_velocity',
        #   'distance', 'gamma', 'heterogeneous_velocity'
        delay_mode='constant',

        # For 'gamma' mode: shape param of pure Gamma delay distribution
        # (delays drawn directly, no distance/velocity factorisation).
        delay_gamma_shape=5.0,

        # For 'heterogeneous_velocity' mode (legacy, graph-hops based):
        # std of Normal-distributed per-edge velocities.
        velocity_std=2.0,

        # For 'matrix_gamma_velocity' mode (Atay & Hutt 2006, Eq. 5.2):
        # Truncated Gamma distribution g(v) ∝ v^(p-1) * exp(-v/q) on (v_lo, v_hi).
        # Mean E[v] = p*q is set to conduction_velocity (q = v_mean / p);
        # higher shape p ⇒ tighter distribution; clip to (v_lo, v_hi).
        velocity_gamma_shape=5.0,         # p (Atay & Hutt suggest p ~ 4-7 for cortex)
        velocity_truncate_low=1.0,        # m/s   (avoid divergent delays)
        velocity_truncate_high=60.0,      # m/s   (biological max for myelinated)

        # Random seed for stochastic delay modes (gamma, heterogeneous_velocity,
        # matrix_gamma_velocity)
        seed=42,
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
        """Compute conduction delay T_ij (seconds) for each directed edge.

        Six delay modes, grouped by what determines the delay:

        ── A. Distance-independent (delay scalar; ignores anatomy) ────────
            'constant'   : All edges share params['delay'] seconds.
                           Use as homogeneous-delay baseline.
            'gamma'      : Delays drawn directly from a Gamma distribution
                           with mean = params['delay'] and shape =
                           params['delay_gamma_shape']. NOT factorised into
                           distance × velocity — pure stochastic delays.

        ── B. Distance-based, single global velocity (Forrester paper) ────
            'matrix'     : T_ij = D[i,j] / v
                              D = params['distance_matrix']      (N×N, mm)
                              v = params['conduction_velocity']  (m/s)
                           Matches Forrester p.7: T_ij = d_ij / v.
                           Use with HCP distances OR with synthetic
                           distances from idealised topologies (e.g.
                           Kavya's src/distance_delays.compute_distance_matrix).

        ── C. Distance-based, heterogeneous per-edge velocity ─────────────
            'matrix_gamma_velocity'
                         : T_ij = D[i,j] / v_ij,  with v_ij ~ truncated
                           Gamma per Atay & Hutt (2006), SIAM J. Appl.
                           Dyn. Syst. 5(4), Eq. 5.2:
                              g(v) ∝ v^(p-1) * exp(-v/q)  on (v_lo, v_hi)
                           shape p   = params['velocity_gamma_shape']
                           scale q   chosen so E[v] = conduction_velocity
                                     (q = v_mean / p)
                           clipped to (velocity_truncate_low,
                                       velocity_truncate_high) m/s.
                           Models myelinated cortico-cortical conduction
                           velocity heterogeneity (Nunez observations,
                           Atay & Hutt Fig 1; peak ~8 m/s, range 0–24).

        ── D. Legacy modes (graph-hops based; dimensionally weak) ─────────
            'distance'   : T_ij = max(graph_hops(j,i), 1) / v.
                           Hops are unitless ⇒ unit of delay is s·hops/m.
                           Useful only as a placeholder for Conti-style
                           abstract topology comparisons; for biophysical
                           setups prefer 'matrix' with explicit distances.
            'heterogeneous_velocity'
                         : Same as 'distance' but per-edge velocities
                           drawn from Normal(v, params['velocity_std'])
                           clipped to ≥ 1 m/s. Same dimensional caveat
                           as 'distance'.

        Notes
        -----
        - All returned delays are in SECONDS.
        - All distance matrices D are expected in MILLIMETRES.
        - All velocities are in METRES PER SECOND.
        - Conversion: T(s) = D(mm) / (v(m/s) * 1000).
        - Random modes (gamma, matrix_gamma_velocity, heterogeneous_velocity)
          use params['seed'] for reproducibility.
        """
        mode = self.p.get('delay_mode', 'constant')
        fixed_delay = self.p['delay']
        v_mean = self.p['conduction_velocity']
        seed = self.p.get('seed', 42)

        if mode == 'constant':
            self.delays = np.full(self.M, fixed_delay)

        elif mode == 'matrix':
            # T_ij = D[i,j] / v   (Forrester p.7)
            D = self.p.get('distance_matrix')
            if D is None:
                raise ValueError("delay_mode='matrix' requires params['distance_matrix']")
            self.delays = np.array([
                D[i, j] / (v_mean * 1000.0)            # mm / (m/s × 1000) = s
                for (j, i) in self.edges
            ])

        elif mode == 'matrix_gamma_velocity':
            # T_ij = D[i,j] / v_ij,  v_ij ~ truncated Gamma
            # Reference: Atay & Hutt (2006), SIAM J. Appl. Dyn. Syst. 5(4), Eq. 5.2.
            D = self.p.get('distance_matrix')
            if D is None:
                raise ValueError(
                    "delay_mode='matrix_gamma_velocity' requires params['distance_matrix']")
            p_shape = self.p.get('velocity_gamma_shape', 5.0)
            v_lo = self.p.get('velocity_truncate_low', 1.0)
            v_hi = self.p.get('velocity_truncate_high', 60.0)
            # Scale q chosen so E[v] = p * q = v_mean
            q_scale = v_mean / p_shape

            rng = np.random.default_rng(seed)
            velocities = rng.gamma(p_shape, q_scale, size=self.M)
            velocities = np.clip(velocities, v_lo, v_hi)

            self.delays = np.array([
                D[i, j] / (vel * 1000.0)               # mm / (m/s × 1000) = s
                for (j, i), vel in zip(self.edges, velocities)
            ])
            # Stash realised per-edge velocities for inspection
            self._gamma_velocities = velocities

        elif mode == 'gamma':
            # Pure Gamma delays (no distance/velocity factorisation)
            k = self.p.get('delay_gamma_shape', 5.0)
            theta = fixed_delay / k                    # mean = k * theta
            rng = np.random.default_rng(seed)
            self.delays = rng.gamma(k, theta, size=self.M)

        elif mode == 'distance':
            # Legacy: graph hops, dimensionally weak
            G = self.network.network
            sp = dict(nx.shortest_path_length(G))
            self.delays = np.array([
                max(sp[j][i], 1) / v_mean
                for (j, i) in self.edges
            ])

        elif mode == 'heterogeneous_velocity':
            # Legacy: graph hops × Normal velocity
            G = self.network.network
            sp = dict(nx.shortest_path_length(G))
            sigma_v = self.p.get('velocity_std', 2.0)
            rng = np.random.default_rng(seed)
            velocities = rng.normal(v_mean, sigma_v, size=self.M)
            velocities = np.maximum(velocities, 1.0)
            self.delays = np.array([
                max(sp[j][i], 1) / vel
                for (j, i), vel in zip(self.edges, velocities)
            ])

        else:
            raise ValueError(f"Unknown delay_mode: {mode}")

        # Universal floor to prevent zero/negative delays (jitcdde requires > 0)
        self.delays = np.maximum(np.array(self.delays, dtype=float), 1e-4)

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
        """Build the symbolic DDE system for jitcdde compilation.

        Constructs 12·N + 2·M symbolic equations: 12 per node (R, V, g, s)
        and 2 per directed edge (g_ij, s_ij driven by R_E_j(t − T_ij)).
        Inter-node coupling is excitatory-only (E → E) and gap junctions
        are intra-population only.

        Returns
        -------
        DDE : jitcdde instance, ready for `.constant_past` and `.integrate`.
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
        """Kuramoto order parameter Z per node (E and I populations).

        Möbius (conformal) map between (R, V) and the Kuramoto unit disk:

            W = π · τ · R + i · V
            Z = (1 − conj(W)) / (1 + conj(W))

        |Z| ∈ [0, 1] measures within-population phase coherence.

        Returns
        -------
        Z_E : ndarray, shape (N, T), complex
        Z_I : ndarray, shape (N, T), complex
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
        """Pairwise Phase-Locking Value (PLV) matrix.

            PLV_ij = | <exp(i · (θ_i(t) − θ_j(t)))>_t |

        where θ_n(t) = arg(Z_n(t)) from compute_Z. Vectorised via
        BLAS matrix product for ~8× speedup at HCP scale.

        Parameters
        ----------
        population : 'E' or 'I'
        t_start : float or None
            Discard transient before this time. Default: 25% of total.

        Returns
        -------
        PLV : ndarray (N, N), real, in [0, 1].
        """
        Z_E, Z_I = self.compute_Z()
        Z = Z_E if population == 'E' else Z_I

        # Discard transient
        if t_start is None:
            t_start = self.time_array[-1] * 0.25
        mask = self.time_array >= t_start
        Z_trimmed = Z[:, mask]

        # e^(iθ) per node per timestep
        exp_itheta = np.exp(1j * np.angle(Z_trimmed))    # (N, T')

        # PLV = | (1/T) Σ_t exp(iθ_i) * conj(exp(iθ_j)) |
        # Matrix form: (E E^†) / T  → BLAS-optimised matmul
        T_eff = exp_itheta.shape[1]
        plv_complex = (exp_itheta @ exp_itheta.conj().T) / T_eff
        PLV = np.abs(plv_complex)
        np.fill_diagonal(PLV, 1.0)
        return PLV

    # ------------------------------------------------------------------
    # Pearson correlation FC (using R_E time series)
    # ------------------------------------------------------------------
    def compute_FC_corr(self, t_start=None):
        """Pearson-correlation FC on raw firing rates R_E(t).

        Operates on the fast neural signal (~10 Hz). NOT directly
        comparable to fMRI BOLD FC — use compute_BOLD_FC for that.

        Parameters
        ----------
        t_start : float or None. Default: 25% of total.

        Returns
        -------
        FC : ndarray (N, N), real, in [-1, 1].
        """
        if t_start is None:
            t_start = self.time_array[-1] * 0.25
        mask = self.time_array >= t_start

        RE_all = self.get_all_RE()[:, mask]  # (N, T')
        FC = np.corrcoef(RE_all)
        return FC

    def compute_BOLD(self, t_start=None):
        """Synthetic BOLD signal per node via Balloon-Windkessel.

        Two layers:
          (1) 4-ODE Balloon-Windkessel state evolution (Forrester Eq 17,
              matches NFESOLVE C++).
          (2) 3-term BOLD output equation
              BOLD = V_0·[k_1(1-q) + k_2(1-q/v) + k_3(1-v)]
              — this output formula is NOT in Forrester or NFESOLVE; it
              is the standard fMRI 3-T form from external references
              (see CLAUDE.md References section).

        Integration: vectorised inline RK4 across all N nodes (same time
        grid as the simulation; piecewise-linear forcing for half-step
        stages). About 6× faster than scipy solve_ivp with equivalent
        accuracy.

        Parameters
        ----------
        t_start : float or None
            Discard transient before this time (s). Default 30 s.

        Returns
        -------
        BOLD : ndarray (N, T'). Post-transient BOLD per node.
        t_bold : ndarray (T',). Time points (s).
        """
        # --- Stephan (2007) 3-T parameters; matches Forrester p.11 ---
        rho       = 0.34   # resting oxygen extraction fraction
        tau_BOLD  = 2.0    # haemodynamic transit time (s)
        k_bw      = 0.65   # rate of signal decay (1/s)
        gamma     = 0.41   # rate of flow-dependent elimination (1/s)
        alpha     = 0.32   # Grubb exponent
        epsilon   = 1.0    # neural-to-vascular coupling (NFESOLVE BW_epsilon)
        V0        = 0.02   # resting blood volume fraction
        k1, k2, k3 = 7.0 * rho, 2.0, 2.0 * rho - 0.2

        t_grid = self.time_array
        T = len(t_grid)
        dt = t_grid[1] - t_grid[0]
        RE_all = self.get_all_RE()                      # (N, T)
        N = self.N

        # Vectorised RHS over all N nodes simultaneously.
        # state shape (4, N) = [x, f, v, q] for each node.
        # R_E shape (N,) at the requested time index.
        inv_alpha = 1.0 / alpha
        one_minus_rho = 1.0 - rho

        def vec_rhs(state, R_E):
            x_, f_, v_, q_ = state                      # each (N,)
            f_safe = np.maximum(f_, 1e-3)
            v_safe = np.maximum(v_, 1e-3)
            dx = epsilon * R_E - k_bw * x_ - gamma * (f_ - 1.0)
            df = x_
            v_pow = v_safe ** inv_alpha
            dv = (f_ - v_pow) / tau_BOLD
            dq = (f_safe / rho * (1.0 - one_minus_rho ** (1.0 / f_safe))
                  - q_ * v_safe ** (inv_alpha - 1.0)) / tau_BOLD
            return np.stack([dx, df, dv, dq])

        # Initial condition: resting baseline (x=0, f=v=q=1) for every node
        state = np.empty((4, N))
        state[0] = 0.0
        state[1] = 1.0
        state[2] = 1.0
        state[3] = 1.0

        # Storage
        x_all = np.empty((N, T)); f_all = np.empty((N, T))
        v_all = np.empty((N, T)); q_all = np.empty((N, T))
        x_all[:, 0], f_all[:, 0], v_all[:, 0], q_all[:, 0] = state

        # --- RK4 integration (vectorised over N nodes) ---
        # R_E is sampled on the same grid as the simulation; for the
        # half-step RK4 stages we use linear interpolation between samples,
        # equivalent to a piecewise-linear forcing assumption.
        for i in range(T - 1):
            R_E_i = RE_all[:, i]
            R_E_h = 0.5 * (RE_all[:, i] + RE_all[:, i + 1])   # midpoint
            R_E_n = RE_all[:, i + 1]
            k1_ = vec_rhs(state,                R_E_i)
            k2_ = vec_rhs(state + 0.5 * dt * k1_, R_E_h)
            k3_ = vec_rhs(state + 0.5 * dt * k2_, R_E_h)
            k4_ = vec_rhs(state + dt * k3_,       R_E_n)
            state = state + (dt / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_)
            # Floor to avoid negative blood volume / oxygenation (numerical safety)
            state[1] = np.maximum(state[1], 1e-3)   # f
            state[2] = np.maximum(state[2], 1e-3)   # v
            state[3] = np.maximum(state[3], 1e-3)   # q

            x_all[:, i + 1], f_all[:, i + 1], v_all[:, i + 1], q_all[:, i + 1] = state

        # --- BOLD output: 3-term Obata (2004) / Stephan (2007), 3-T form ---
        BOLD_all = V0 * (k1 * (1.0 - q_all)
                         + k2 * (1.0 - q_all / v_all)
                         + k3 * (1.0 - v_all))

        if t_start is None:
            t_start = 30.0
        mask = t_grid >= t_start
        return BOLD_all[:, mask], t_grid[mask]

    def compute_BOLD_FC(self, t_start=None, enigma_aligned=False):
        """Pearson-correlation FC from synthetic BOLD (Balloon-Windkessel).

        Correct comparison target for fMRI BOLD empirical FC.

        Parameters
        ----------
        t_start : float or None. Default 30 s.
        enigma_aligned : bool, default False.
            If True, apply the ENIGMA HCP per-subject preprocessing pipeline
            (Royer et al. 2022, ENIGMA Toolbox docs):
              (i)   Pearson correlation
              (ii)  negative values clipped to 0
              (iii) Fisher-z transform: z = arctanh(r)
            Group-averaging across subjects (step iv in ENIGMA) is left to the
            caller (e.g. multi-seed ensemble in the experiment script).
            Output range becomes [0, +inf) in z-space, matching empirical.

        Returns
        -------
        FC_BOLD : ndarray (N, N).  Raw mode: real in [-1, 1]. Aligned mode:
                  in [0, +inf), z-space, with negatives thresholded to 0.
        """
        BOLD, _ = self.compute_BOLD(t_start=t_start)
        FC = np.corrcoef(BOLD)
        if enigma_aligned:
            FC = np.maximum(FC, 0.0)
            FC = np.arctanh(np.clip(FC, 0.0, 1.0 - 1e-9))
            np.fill_diagonal(FC, 0.0)            # ENIGMA convention
        return FC

    def compute_dynamic_PLV(self, window_sec=10.0, overlap=0.9,
                            population='E', t_start=None):
        """Dynamic FC via sliding-window PLV.

        Forrester Fig 7 setup: 10 s window, 90% overlap.

        Parameters
        ----------
        window_sec : float
        overlap : float in [0, 1)
        population : 'E' or 'I'
        t_start : float or None. Default 20% of total.

        Returns
        -------
        dFC : ndarray (n_windows, N, N).
        t_centers : ndarray (n_windows,). Window centre times (s).
        """
        Z_E, Z_I = self.compute_Z()
        Z = Z_E if population == 'E' else Z_I
        t = self.time_array

        if t_start is None:
            t_start = t[-1] * 0.2
        dt = t[1] - t[0]
        win_samples = int(window_sec / dt)
        step_samples = int(win_samples * (1 - overlap))
        if step_samples < 1:
            step_samples = 1

        start_idx = int(t_start / dt)
        end_idx = len(t)

        windows = []
        centers = []
        idx = start_idx
        while idx + win_samples <= end_idx:
            Z_win = Z[:, idx:idx + win_samples]      # (N, win_samples)
            # Vectorised PLV via matrix product (same as compute_PLV)
            exp_itheta = np.exp(1j * np.angle(Z_win))
            PLV_win = np.abs(exp_itheta @ exp_itheta.conj().T) / win_samples
            np.fill_diagonal(PLV_win, 1.0)
            windows.append(PLV_win)
            centers.append(t[idx + win_samples // 2])
            idx += step_samples

        return np.array(windows), np.array(centers)