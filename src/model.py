"""
Wilson-Cowan Delay Differential Equation (DDE) Metapopulation Model

Implements the Wilson-Cowan DDE model from Conti & Van Gorder (2019)
using jitcdde for symbolic DDE integration.

Equations for node n = 0, ..., N-1:

  (1/T_e) * dE_n/dt = -E_n + S(c_ee*E_n(t-tau1) + c_ie*I_n(t-tau2) + P
                            + k * sum_j W_nj * E_j(t-rho))

  (1/(T_i*alpha)) * dI_n/dt = -I_n + S(c_ei*E_n(t-tau2) + c_ii*I_n(t-tau1) + Q)

  S(x) = 1 / (1 + exp(-beta * x))

State ordering: [E_0, E_1, ..., E_{N-1}, I_0, I_1, ..., I_{N-1}]
"""

import numpy as np
from jitcdde import jitcdde, y, t
from symengine import exp


# ---------------------------------------------------------------------------
# Default parameters (Conti & Van Gorder 2019)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "mode": "wilson_cowan",
    # Synaptic weights
    "c_ee": 1.0,
    "c_ei": 1.0,
    "c_ie": -1.0,
    "c_ii": -1.0,
    # Delays
    "tau_1": 1.0, # intra-node same-type delay (E->E, I->I)
    "tau_2": 1.4, # intra-node cross-type delay (E->I, I->E)
    "rho": 10.0, # Inter-node E to E coupling delay
    # Sigmoid
    "beta": 10.0,
    # Timescales
    "alpha": 0.6,
    "T_e": 1.0,
    "T_i": 1.0,
    # External inputs
    "P": 0.5,
    "Q": 0.5,
    # Coupling strength
    "k": 1.0, # can compare k = 1 vs k = 10
}


def _sigmoid(x, beta):
    """Symbolic sigmoid function for use in jitcdde equations."""
    return 1.0 / (1.0 + exp(-beta * x))


class Model:
    """Wilson-Cowan DDE metapopulation model solved with jitcdde."""

    def __init__(self, network, params=None):
        self.network = network
        self.params = {**DEFAULT_PARAMS}
        if params is not None:
            self.params.update(params)
        self.time_array = None # Time grid
        self.trajectories = None  # Reulsting simulated trajectories 2xNxT
        self.process_mode()

    def process_mode(self):
        self.mode = self.params['mode']
        if self.mode == 'wilson_cowan':
            self.C = 2  # Number of model components
            self.components = ['Excitatory', 'Inhibitory']

    # ------------------------------------------------------------------
    # Time grid
    # ------------------------------------------------------------------
    def set_time_grid(self, duration=1000, dt=0.1):
        """Define the integration time grid."""
        self.duration = duration
        self.dt = dt
        self.time_array = np.arange(0, duration, dt)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    def set_initial_conditions(self, N):
        """Placeholder kept for API compatibility with Metapopulation.
        Actual initial conditions are set inside run() via jitcdde's
        constant_past mechanism.
        """
        pass

    # ------------------------------------------------------------------
    # Build the jitcdde system
    # ------------------------------------------------------------------
    def wilson_cowan(self):
        """Construct and return the jitcdde DDE system for the
        Wilson-Cowan metapopulation model.

        Returns
        -------
        DDE : jitcdde instance, ready for past-setting and integration.
        """
        p = self.params
        N = self.network.N
        W = self.network.A  # adjacency matrix (N x N)

        # Unpack parameters
        c_ee = p["c_ee"]
        c_ei = p["c_ei"]
        c_ie = p["c_ie"]
        c_ii = p["c_ii"]
        tau1 = p["tau_1"]
        tau2 = p["tau_2"]
        rho = p["rho"]
        beta = p["beta"]
        alpha = p["alpha"]
        T_e = p["T_e"]
        T_i = p["T_i"]
        P = p["P"]
        Q = p["Q"]
        k = p["k"]

        # Build symbolic RHS for 2N state variables
        f = []

        # --- Excitatory equations: indices 0 .. N-1 ---
        for n in range(N):
            # Intra-node terms (delayed)
            ee_term = c_ee * y(n, t - tau1)          # E_n(t - tau1)
            ie_term = c_ie * y(N + n, t - tau2)      # I_n(t - tau2)

            # Inter-node coupling: k * sum_j W_nj * E_j(t - rho)
            coupling = sum(
                W[n, j] * y(j, t - rho)
                for j in range(N)
                if W[n, j] != 0
            )

            sigmoid_arg = ee_term + ie_term + P + k * coupling
            dEn_dt = T_e * (-y(n) + _sigmoid(sigmoid_arg, beta))
            f.append(dEn_dt)

        # --- Inhibitory equations: indices N .. 2N-1 ---
        for n in range(N):
            ei_term = c_ei * y(n, t - tau2)          # E_n(t - tau2)
            ii_term = c_ii * y(N + n, t - tau1)      # I_n(t - tau1)

            sigmoid_arg = ei_term + ii_term + Q
            dIn_dt = (T_i * alpha) * (-y(N + n) + _sigmoid(sigmoid_arg, beta))
            f.append(dIn_dt)

        # Create the jitcdde system
        DDE = jitcdde(f)
        return DDE

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------
    def run(self, initial_conditions=None):
        """Build the DDE system, set initial conditions, integrate,
        and store results in self.trajectories (shape 2 x N x T)
        and self.time_array.

        Parameters
        ----------
        initial_conditions : array-like, optional
            Flat array of length 2*N for [E_0,...,E_{N-1}, I_0,...,I_{N-1}].
            Defaults to 0.5 for all variables.
        """
        N = self.network.N
        n_vars = 2 * N

        # Build the DDE
        DDE = self.wilson_cowan()

        # Initial conditions
        if initial_conditions is not None:
            initial = np.asarray(initial_conditions, dtype=float)
        else:
            initial = np.full(n_vars, 0.5)
        DDE.constant_past(initial)

        # Use adjust_diff to keep integration starting near t=0
        # so transient dynamics are visible (step_on_discontinuities
        # advances past all delays, hiding the initial transient).
        DDE.adjust_diff()

        # Integrate over the time grid
        results = np.empty((len(self.time_array), n_vars))
        for i, t_val in enumerate(self.time_array):
            if t_val <= DDE.t:
                results[i, :] = DDE.integrate(DDE.t)
            else:
                results[i, :] = DDE.integrate(t_val)

        # Reshape into (2, N, T): [Excitatory, Inhibitory] x nodes x time
        E = results[:, :N].T       # (N, T)
        I = results[:, N:].T       # (N, T)
        self.trajectories = np.stack([E, I], axis=0)  # (2, N, T)
