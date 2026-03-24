"""
Next Generation Neural Mass Model (single node).

Implements the 12-ODE mean-field model from:
  Forrester et al. (2024) -- Eqs 7-9, parameters from p.7

The model describes an E-I pair derived from the Ott-Antonsen ansatz 
reduction of networks of quadratic integrate-and-fire (QIF) neurons.

State variables (12):
    R_E, R_I  -- population firing rates
    V_E, V_I  -- mean membrane potentials
    g_EE, g_EI, g_IE, g_II  -- synaptic conductances
    s_EE, s_EI, s_IE, s_II  -- synaptic gating variables
"""

import numpy as np
from scipy.integrate import solve_ivp


# State ordering: [R_E, R_I, V_E, V_I, g_EE, g_EI, g_IE, g_II, s_EE, s_EI, s_IE, s_II]
IDX_RE, IDX_RI = 0, 1
IDX_VE, IDX_VI = 2, 3
IDX_GEE, IDX_GEI, IDX_GIE, IDX_GII = 4, 5, 6, 7
IDX_SEE, IDX_SEI, IDX_SIE, IDX_SII = 8, 9, 10, 11


def default_params():
    """Return default Next Gen network parameters (Forrester et al. 2024, p.7)."""
    return dict(
        # Mean population inputs
        eta_I=3.0,
        eta_E=-2.5,

        # Input distributions’ widths at half maximum 
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
        
    )


class NextGenModel:
    """Single-node Next Generation Neural Mass Model (12 ODEs)."""

    def __init__(self, params=None):
        """
        Parameters
        ----------
        params : dict or None
            Model parameters.  Any keys not supplied fall back to
            ``default_params()``.
        """
        self.p = default_params()
        if params is not None:
            self.p.update(params)

        # Storage filled by run()
        self.t = None
        self.state = None

    # ------------------------------------------------------------------
    # Right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, state):
        """Compute the 12 derivatives.

        Parameters
        ----------
        t : float
            Current time (unused for autonomous system, kept for solve_ivp).
        state : array, shape (12,)
            Current state vector.

        Returns
        -------
        dstate : array, shape (12,)
        """
        p = self.p

        # state
        R_E, R_I = state[IDX_RE], state[IDX_RI]
        V_E, V_I = state[IDX_VE], state[IDX_VI]
        g_EE, g_EI = state[IDX_GEE], state[IDX_GEI]
        g_IE, g_II = state[IDX_GIE], state[IDX_GII]
        s_EE, s_EI = state[IDX_SEE], state[IDX_SEI]
        s_IE, s_II = state[IDX_SIE], state[IDX_SII]
        
        # parameters
        eta_E, eta_I = p['eta_E'], p['eta_I']
        delta_E, delta_I = p['delta_E'], p['delta_I']
        tau_E, tau_I = p['tau_E'], p['tau_I']
        alpha_EE, alpha_EI, alpha_IE, alpha_II = p['alpha_EE'], p['alpha_EI'], p['alpha_IE'], p['alpha_II']
        kappa_v_EE, kappa_v_EI, kappa_v_IE, kappa_v_II = p['kappa_v_EE'], p['kappa_v_EI'], p['kappa_v_IE'], p['kappa_v_II']
        v_syn_EE, v_syn_EI, v_syn_IE, v_syn_II = p['v_syn_EE'], p['v_syn_EI'], p['v_syn_IE'], p['v_syn_II']
        kappa_s_EE, kappa_s_EI, kappa_s_IE, kappa_s_II = p['kappa_s_EE'], p['kappa_s_EI'], p['kappa_s_IE'], p['kappa_s_II']
 

        # --- Firing-rate equations (Eq 7)---
        # Sum of conductances + gap-junction terms sum_b (g_ab + kappa_v_ab)
        sum_g_kv_E = (g_EE + kappa_v_EE) + (g_EI + kappa_v_EI)
        sum_g_kv_I = (g_IE + kappa_v_IE) + (g_II + kappa_v_II)

        dR_E = (-R_E * sum_g_kv_E + 2.0 * R_E * V_E + delta_E / (np.pi * tau_E)) / tau_E
        dR_I = (-R_I * sum_g_kv_I + 2.0 * R_I * V_I + delta_I / (np.pi * tau_I)) / tau_I

        # --- Mean-voltage equations (Eq 8) ---
        # Synaptic current contributions:  sum_b g_ab * (v_syn_ab - V_a)
        I_syn_E = (g_EE * (v_syn_EE - V_E) + g_EI * (v_syn_EI - V_E))
        I_syn_I = (g_IE * (v_syn_IE - V_I) + g_II * (v_syn_II - V_I))

        # Gap-junction current contributions:  sum_b kappa_v_ab * (V_b - V_a)
        I_gap_E = (kappa_v_EE * (V_E - V_E) + kappa_v_EI * (V_I - V_E))
        I_gap_I = (kappa_v_IE * (V_E - V_I) + kappa_v_II * (V_I - V_I))

        dV_E = (eta_E + V_E ** 2 - (np.pi * tau_E * R_E) ** 2
                + I_syn_E + I_gap_E) / tau_E
        dV_I = (eta_I + V_I ** 2 - (np.pi * tau_I * R_I) ** 2
                + I_syn_I + I_gap_I) / tau_I

        # --- Synaptic conductance equations (Eq 9) ---
        dg_EE = alpha_EE * (s_EE - g_EE)
        dg_EI = alpha_EI * (s_EI - g_EI)
        dg_IE = alpha_IE * (s_IE - g_IE)
        dg_II = alpha_II * (s_II - g_II)

        # --- Synaptic gating equations (Eq 9) ---
        ds_EE = alpha_EE * (kappa_s_EE * R_E - s_EE)
        ds_EI = alpha_EI * (kappa_s_EI * R_I - s_EI)
        ds_IE = alpha_IE * (kappa_s_IE * R_E - s_IE)
        ds_II = alpha_II * (kappa_s_II * R_I - s_II)

        return np.array([
            dR_E, dR_I, dV_E, dV_I,
            dg_EE, dg_EI, dg_IE, dg_II,
            ds_EE, ds_EI, ds_IE, ds_II,
        ])

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def run(self, duration=1.0, dt=0.001, y0=None):
        """Integrate the model using RK45.

        Parameters
        ----------
        duration : float
            Simulation length in seconds.
        dt : float
            Maximum output time-step (also used as max_step for the solver).
        y0 : array-like or None
            Initial conditions (length 12).  If None, use defaults.

        Returns
        -------
        t : ndarray
            Time points.
        state : ndarray, shape (12, N)
            State trajectories.
        """
        if y0 is None:
            y0 = np.array([
                0.1, 0.1,       # R_E, R_I
                -2.0, -2.0,     # V_E, V_I
                0.01, 0.01, 0.01, 0.01,  # g_EE, g_EI, g_IE, g_II
                0.01, 0.01, 0.01, 0.01,  # s_EE, s_EI, s_IE, s_II
            ])

        t_span = (0.0, duration)
        t_eval = np.arange(0.0, duration, dt)

        sol = solve_ivp(
            self._rhs, t_span, y0,
            method='RK45',
            t_eval=t_eval,
            max_step=dt,
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        self.t = sol.t
        self.state = sol.y  # shape (12, N)
        return self.t, self.state

    # ------------------------------------------------------------------
    # Kuramoto order parameter
    # ------------------------------------------------------------------
    def compute_Z(self):
        """Compute the Kuramoto order parameter Z for E and I populations.

        Uses the conformal map:
            W = pi * tau * R + i * V
            Z = (1 - conj(W)) / (1 + conj(W))

        Returns
        -------
        Z_E, Z_I : ndarray (complex)
            Order parameter time series.  |Z| in [0,1] measures
            within-population synchrony.
        """
        if self.state is None:
            raise RuntimeError("Call run() before compute_Z().")

        p = self.p
        R_E, R_I = self.state[IDX_RE], self.state[IDX_RI]
        V_E, V_I = self.state[IDX_VE], self.state[IDX_VI]

        W_E = np.pi * p['tau_E'] * R_E + 1j * V_E
        W_I = np.pi * p['tau_I'] * R_I + 1j * V_I
        
        # Mobius transformation
        Z_E = (1.0 - np.conj(W_E)) / (1.0 + np.conj(W_E))
        Z_I = (1.0 - np.conj(W_I)) / (1.0 + np.conj(W_I))

        return Z_E, Z_I

