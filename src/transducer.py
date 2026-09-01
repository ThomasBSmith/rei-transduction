"""
Input-output machinery for the microwave-optical transducer.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Transducer:
    """
    Model for a microwave-optical transducer.

    Attributes
    ----------
    kappa_a : float
        Optical linewidth.
    kappa_b : float
        Microwave linewidth.
    omega_b : float
        Microwave frequency.
    Delta : float
        Optical detuning (Delta = omega_b for resonant transduction).
    eta_a : float
        Optical external coupling fraction kappa_a^ext / kappa_a.
    eta_b : float
        Microwave external coupling fraction kappa_b^ext / kappa_b.
    """
    kappa_a : float
    kappa_b : float
    omega_b : float
    Delta : float
    eta_a : float
    eta_b : float

    @classmethod
    def from_params(cls, params, **overrides):
        """
        Build a Transducer from a parameter dict, ignoring keys that are not
        fields of the dataclass (e.g. Delta_a, T). `overrides` set or replace
        fields, e.g. Delta = ±omega_b for the transduction / squeezing resonance.
        """
        fields = {k: params[k] for k in cls.__dataclass_fields__ if k in params}
        return cls(**{**fields, **overrides})

    def A(self, g_t, g_s):
        """
        System matrix A.

        Arguments
        ---------
        g_t : complex
            Transduction (beam-splitter) coupling rate.
        g_s : complex
            Parasitic two-mode squeezing coupling rate.
        """
        return np.array([
            [-1j*self.Delta - self.kappa_a/2,   1j*g_t,                                 0,                                  1j*g_s                              ],
            [ 1j*np.conj(g_t),                 -1j*self.omega_b - self.kappa_b/2,       1j*g_s,                             0                                   ],
            [ 0,                               -1j*np.conj(g_s),                        1j*self.Delta - self.kappa_a/2,    -1j*np.conj(g_t)                     ],
            [-1j*np.conj(g_s),                  0,                                     -1j*g_t,                             1j*self.omega_b - self.kappa_b/2    ]
            ])

    def scattering(self, omega, g_t, g_s):
        """
        Scattering matrix S(omega) and internal-loss matrix L(omega).

        Arguments
        ---------
        omega : float
            Frequency at which to evaluate the scattering matrices.
        g_t : complex
            Transduction (beam-splitter) coupling rate.
        g_s : complex
            Parasitic squeezing coupling rate.
        """

        # External loss rates.
        kappa_a_ext = self.eta_a * self.kappa_a
        kappa_b_ext = self.eta_b * self.kappa_b

        # Internal loss rates.
        kappa_a_int = self.kappa_a - kappa_a_ext
        kappa_b_int = self.kappa_b - kappa_b_ext

        # External and internal loss matrices.
        K_ext = np.diag([kappa_a_ext, kappa_b_ext,]*2)
        K_int = np.diag([kappa_a_int, kappa_b_int,]*2)

        # Inverse of the system matrix A plus the frequency term.
        R = np.linalg.inv(self.A(g_t, g_s) + 1j*omega * np.eye(4))

        # Scattering matrix S and internal-loss matrix L.
        S = -np.eye(4) - np.sqrt(K_ext) @ R @ np.sqrt(K_ext)
        L = -np.sqrt(K_ext) @ R @ np.sqrt(K_int)
        return S, L

    def channel(self, omega, g_t, g_s):
        """
        Phase-corrected scattering (S, L) at frequency omega.
        
        Arguments
        ---------
        omega : float
            Frequency at which to evaluate the scattering matrices.
        g_t : complex
            Transduction (beam-splitter) coupling rate.
        g_s : complex
            Parasitic squeezing coupling rate.
        """
        return phase_correct(*self.scattering(omega, g_t, g_s))


def phase_correct(S, L):
    """
    Fix the output-mode phase reference so that S_12 is real and positive.

    Arguments
    ---------
    S : ndarray
        Scattering matrix.
    L : ndarray
        Internal-loss matrix.
    """

    # Extract phase of the transduction term.
    phase = np.exp(-1j * np.angle(S[0, 1]))

    # Copy S and L matrices.
    S = S.copy()
    L = L.copy()

    # Phase-correct the first row.
    S[0, :] *= phase
    L[0, :] *= phase

    # Return phase-corrected matrices.
    return S, L


def aout_moments(S, L):
    """
    Extract the signal transfer and the noise moments of the optical output.

    Arguments
    ---------
    S : ndarray
        Phase-corrected scattering matrix.
    L : ndarray
        Phase-corrected internal-loss matrix.

    Returns
    -------
        S_12                                            (transduction amplitude),
        S_14                                            (two-mode squeezing amplitude),
        expect_NdN = <N† N>                             (added photon number),
        expect_NN = <N N>                               (anomalous noise moment),
        comm_NNd = [N, N†] = 1 - |S_12|^2 + |S_14|^2    (noise commutator).
    """

    # Transduction and squeezing amplitudes.
    S_12 = S[0, 1]
    S_14 = S[0, 3]

    # Second moments of N over the vacuum ports.
    expect_NdN = np.abs(S[0, 2])**2 + np.abs(L[0, 2])**2 + np.abs(L[0, 3])**2
    expect_NN  = S[0, 0]*S[0, 2] + L[0, 0]*L[0, 2] + L[0, 1]*L[0, 3]

    # Noise commutator.
    comm_NNd = 1 - np.abs(S_12)**2 + np.abs(S_14)**2

    return S_12, S_14, expect_NdN, expect_NN, comm_NNd


def measured_efficiency(nbar, S_12, S_14, expect_NdN, b_sq=0.0):
    """
    Photon-number conversion efficiency measured at the optical output:

        eta_meas = <a_out† a_out> / n̄
                 = |S_12|²                          [eta, coherent conversion]
                    + |S_14|² (1 + 1/n̄) + <N†N>/n̄   [G_0, phase-insensitive gain]
                    - (2/n̄) Re(S_14* S_12 <b_in²>)  [G_phi, phase-sensitive gain]

    Arguments
    ---------
    nbar : float
        Mean photon number of the microwave input state.
    S_12, S_14 : complex
        Transduction and two-mode-squeezing amplitudes (aout_moments).
    expect_NdN : complex
        Added-noise photon number <N†N> (aout_moments).
    b_sq : complex
        Second moment <b_in²> of the input state.
    """
    return (
        np.abs(S_12)**2
        + np.abs(S_14)**2 * (1 + 1/nbar)
        + np.real(expect_NdN)/nbar
        - 2/nbar * np.real(np.conj(S_14) * S_12 * b_sq)
        )
