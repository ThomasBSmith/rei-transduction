"""Input-output machinery for the microwave-optical transducer."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Transducer:
    """
    A microwave-optical transducer at the transduction resonance (Δ' = ω_b').

    Attributes
    ----------
    kappa_a : float
        Optical linewidth.
    kappa_b : float
        Microwave linewidth.
    omega_b : float
        Microwave frequency.
    Delta : float
        Optical detuning (Δ' = ω_b' on the transduction resonance).
    eta_a : float
        Optical external coupling fraction kappa_a^ext / kappa_a.
    eta_b : float
        Microwave external coupling fraction kappa_b^ext / kappa_b.
    """
    kappa_a: float
    kappa_b: float
    omega_b: float
    Delta: float
    eta_a: float = 1.0
    eta_b: float = 1.0

    def A(self, g_t, g_s):
        """
        System matrix A at the transduction resonance (Δ' = ω_b').

        Arguments
        ---------
        g_t : complex
            Transduction (beam-splitter) coupling rate.
        g_s : complex
            Parasitic squeezing coupling rate.
        """
        ka, kb, wb, D = self.kappa_a, self.kappa_b, self.omega_b, self.Delta
        return np.array([
            [-1j*D - ka/2,       1j*g_t,             0,                  1j*g_s          ],
            [ 1j*np.conj(g_t),  -1j*wb - kb/2,       1j*g_s,             0               ],
            [ 0,                -1j*np.conj(g_s),    1j*D - ka/2,       -1j*np.conj(g_t) ],
            [-1j*np.conj(g_s),   0,                 -1j*g_t,             1j*wb - kb/2    ]
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
        ka, kb, ea, eb = self.kappa_a, self.kappa_b, self.eta_a, self.eta_b

        # External and internal loss matrices.
        K_ext = np.diag([ea*ka, eb*kb, ea*ka, eb*kb])
        K_int = np.diag([(1-ea)*ka, (1-eb)*kb, (1-ea)*ka, (1-eb)*kb])

        # Inverse of the system matrix A plus the frequency term.
        R = np.linalg.inv(self.A(g_t, g_s) + 1j*omega*np.eye(4))

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

    The deterministic transduction phase is a free choice of reference, so we
    rotate the output optical mode a_out -> exp(-i arg S_12) a_out, which rescales
    the whole first row of S and L. Only the phase of S_14 relative to S_12
    survives.

    Arguments
    ---------
    S : ndarray
        Scattering matrix.
    L : ndarray
        Internal-loss matrix.
    """
    phase = np.exp(-1j * np.angle(S[0, 1]))
    S = S.copy()
    L = L.copy()
    S[0, :] *= phase
    L[0, :] *= phase
    return S, L


def aout_moments(S, L):
    """
    Extract the signal transfer and the noise moments of the added-noise operator
    N from the first row of the (phase-corrected) scattering matrices.

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
        expect_NdN  = <N† N>                            (added photon number),
        expect_NN   = <N N>                             (anomalous noise moment),
        comm_NNd = [N, N†] = 1 - |S_12|² + |S_14|²      (noise commutator).
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
