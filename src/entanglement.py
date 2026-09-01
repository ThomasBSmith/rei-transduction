"""
Tools to compute output covariance and logarithmic negativity.
"""

import numpy as np
from scipy.integrate import quad_vec

# Ladder -> quadrature transformation [4 x 4].
M = (1/np.sqrt(2)) * np.array([
    [1,   0,  1,  0],
    [0,   1,  0,  1],
    [-1j, 0,  1j, 0],
    [0,  -1j, 0,  1j]
    ])

# Partial-transpose matrix [4 x 4].
P = np.diag([1, 1, 1, -1])

# Symplectic form [4 x 4].
OMEGA = np.block([
    [np.zeros((2, 2)), np.eye(2)],
    [-np.eye(2), np.zeros((2, 2))]
    ])


def filter_matrix(omega, omega_b, tau):
    """
    Gaussian temporal-mode filter matrix F(omega) of width tau, centred on the
    optical (-omega_b) and microwave (+omega_b) sidebands.

    Arguments
    ---------
    omega : float
        Frequency at which to evaluate the filter.
    omega_b : float
        Microwave frequency (sideband location).
    tau : float
        Temporal mode width.
    """

    # Gaussian filter function.
    f = lambda x: (tau**2/np.pi)**0.25 * np.exp(-(tau*x)**2/2)

    # Diagonal filter matrix (+,-,-,+) comes from definition of Fourier transform. 
    return np.diag([
        f(omega + omega_b),
        f(omega - omega_b),
        f(omega - omega_b),
        f(omega + omega_b)
        ])


def spectral_covariance(transducer, omega, g_t, g_s):
    """
    Spectral covariance matrix Sigma(omega) = ½(S S† + L L†) of the output field,
    with all input modes in vacuum.

    Arguments
    ---------
    transducer : Transducer
        Transducer at the entanglement resonance (Delta = -omega_b).
    omega : float
        Frequency at which to evaluate the covariance.
    g_t : float
        Beam-splitter interaction strength.
    g_s : float
        Two-mode squeezing interaction strength.
    """
    S, L = transducer.scattering(omega, g_t, g_s)
    return 0.5*(S @ S.conj().T + L @ L.conj().T)


def covariance(transducer, g_t, g_s, tau):
    """
    Quadrature covariance matrix V = ∫ M F(omega) Sigma(omega) F(omega) M† domega,
    integrated over the two sidebands (F is negligible elsewhere for this width).

    Arguments
    ---------
    transducer : Transducer
        Transducer at the entanglement resonance (Delta = -omega_b).
    g_t : float
        Beam-splitter interaction strength.
    g_s : float
        Two-mode squeezing interaction strength.
    tau : float
        Temporal mode width.
    """

    # Extract microwave cavity frequency.
    omega_b = transducer.omega_b

    # Compose integrand.
    integrand = lambda omega: (
        M @ filter_matrix(omega, omega_b, tau)
        @ spectral_covariance(transducer, omega, g_t, g_s)
        @ filter_matrix(omega, omega_b, tau) @ M.conj().T
        ).real

    # Set the integration edge at many multiples of tau.
    edge = 20/tau

    # Check that integration regions don't overlap.
    assert edge < omega_b, "Sideband integration windows overlap."

    # Integrate the two sidebands.
    V_neg, _ = quad_vec(integrand, -omega_b - edge, -omega_b + edge, limit=200)
    V_pos, _ = quad_vec(integrand,  omega_b - edge,  omega_b + edge, limit=200)

    # Return the computed covariance.
    return V_neg + V_pos


def log_negativity(V):
    """
    Logarithmic negativity of a two-mode Gaussian state from its covariance matrix
    V, via the minimum symplectic eigenvalue of the partial transpose.

    Arguments
    ---------
    V : ndarray
        Quadrature covariance matrix.
    """
    nu_min = np.min(np.abs(np.linalg.eigvals(1j*OMEGA @ (P @ V @ P))))
    return max(0.0, -np.log2(2*nu_min))
