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
        print("Model parameters:")
        print(self.params)
        self.time_array = None
        self.trajectories = None
        self.process_mode()

    def process_mode(self):
        self.mode = self.params['mode']
        if self.mode in ['wilson_cowan', 'wilson_cowan_efficient']:
            self.C = 2
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
    def set_initial_conditions(self, initial_conditions=None):
        """Set initial conditions of the simulation.
        Either pass a constant, or pass a dictionary to set different conditions.
        """
        N = self.network.N
        if initial_conditions is None:
            initial = np.full(2*N, 0.5)
        elif isinstance(initial_conditions, float):
            initial = np.full(2*N, initial_conditions)
        elif isinstance(initial_conditions, dict):
            e_val = initial_conditions['E']
            i_val = initial_conditions['I']
            initial = np.full(2*N, e_val)
            initial[N:] = i_val
        self.initial_conditions = initial

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
        print("Building Wilson-Cowan DDE system with jitcdde...")
        p = self.params
        N = self.network.N
        W = self.network.A

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

        f = []

        # --- Excitatory equations: indices 0 .. N-1 ---
        for n in range(N):
            ee_term = c_ee * y(n, t - tau1)
            ie_term = c_ie * y(N + n, t - tau2)

            coupling = sum(
                W[n, j] * y(j, t - (rho if np.isscalar(rho) else rho[n, j]))
                for j in range(N)
                if W[n, j] != 0
            )
            P_n = P if np.isscalar(P) else P[n]
            sigmoid_arg = ee_term + ie_term + P_n + k * coupling
            dEn_dt = T_e * (-y(n) + _sigmoid(sigmoid_arg, beta))
            f.append(dEn_dt)

        # --- Inhibitory equations: indices N .. 2N-1 ---
        for n in range(N):
            ei_term = c_ei * y(n, t - tau2)
            ii_term = c_ii * y(N + n, t - tau1)

            sigmoid_arg = ei_term + ii_term + Q
            dIn_dt = (T_i * alpha) * (-y(N + n) + _sigmoid(sigmoid_arg, beta))
            f.append(dIn_dt)

        DDE = jitcdde(f)
        return DDE

    def wilson_cowan_efficient(self):
        """Fast Vectorized Euler integration for the Wilson-Cowan model."""
        print("Using Wilson-Cowan efficient solver...")
        p = self.params  
        N = self.network.N
        W = self.network.A
        
        dt = self.time_array[1] - self.time_array[0] if len(self.time_array) > 1 else 0.1
        steps = len(self.time_array)

        # 1. Calculate discrete delay steps
        t1 = max(1, int(p['tau_1'] / dt))
        t2 = max(1, int(p['tau_2'] / dt))
        tr = max(1, int(p['rho'] / dt))
        max_delay = max(t1, t2, tr)

        # 2. Allocate total memory: history buffer + active simulation
        total_steps = steps + max_delay
        E = np.full((total_steps, N), self.initial_conditions[:N])
        I = np.full((total_steps, N), self.initial_conditions[N:])

        c_ee, c_ie, c_ei, c_ii = p['c_ee'], p['c_ie'], p['c_ei'], p['c_ii']
        P, Q, beta, k, alpha = p['P'], p['Q'], p['beta'], p['k'], p['alpha']
        T_e, T_i = p['T_e'], p['T_i']

        for i in range(max_delay, total_steps - 1):
            coupling = W @ E[i - tr]
            e_arg = c_ee * E[i - t1] + c_ie * I[i - t2] + P + k * coupling
            i_arg = c_ei * E[i - t2] + c_ii * I[i - t1] + Q
            
            dE = T_e * (-E[i] + 1.0 / (1.0 + np.exp(-beta * np.clip(e_arg, -10, 10))))
            dI = (T_i * alpha) * (-I[i] + 1.0 / (1.0 + np.exp(-beta * np.clip(i_arg, -10, 10))))
            
            E[i+1] = E[i] + dE * dt
            I[i+1] = I[i] + dI * dt

        E_out = E[max_delay:]
        I_out = I[max_delay:]

        return np.hstack((E_out, I_out))
        
    def run(self):
        """Build the DDE system, set initial conditions, integrate,
        and store results in self.trajectories (shape 2 x N x T)
        and self.time_array.
        Additional new efficient mode run
        """
        N = self.network.N

        if self.mode == "wilson_cowan_efficient":
            results = self.wilson_cowan_efficient()
        else:
            DDE = self.wilson_cowan()
            DDE.constant_past(self.initial_conditions)
            DDE.compile_C(simplify=False)
            DDE.adjust_diff()

            results = np.empty((len(self.time_array), 2*N))
            for i, t_val in enumerate(self.time_array):
                if t_val < DDE.t:
                    results[i, :] = self.initial_conditions
                else:
                    results[i, :] = DDE.integrate(t_val)

        E = results[:, :N].T
        I = results[:, N:].T
        self.trajectories = np.stack([E, I], axis=0)