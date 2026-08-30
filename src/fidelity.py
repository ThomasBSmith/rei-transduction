"""
Tools for calculating state fidelities in phase-space using characteristic functions (App. F1).
"""

import numpy as np
from scipy.special import eval_laguerre


def Phi_fock(z, n):
    """
    Characteristic function of the number state |n>.

    Arguments
    ---------
    z : complex
        Complex argument of the characteristic function.
    n : int
        Fock state number.
    """
    return np.exp(-np.abs(z)**2/2) * eval_laguerre(n, np.abs(z)**2)


def Phi_coherent(z, alpha):
    """
    Characteristic function of the coherent state |alpha>.

    Arguments
    ---------
    z : complex
        Complex argument of the characteristic function.
    alpha : complex
        Coherent state amplitude.
    """
    return np.exp(-np.abs(z)**2/2) * np.exp(z*np.conj(alpha) - np.conj(z)*alpha)


def fidelity(Phi_in, S_12, S_14, expect_NdN, expect_NN, comm_NNd, R=10.0, N=501):
    """
    Transduction fidelity F = (1/π) ∫ d²z Φ_in(z) Φ_out(-z) (App. F1, Eq. app-eqn:fidelity),
    evaluated on a square grid in the complex plane. The output characteristic function is

        Φ_out(-z) = Φ_in(-z̃) exp{ ½[ z² <N†²> + z*² <N²> - |z|²(2<N†N> + [N,N†]) ] },

    with the rescaled argument z̃ = z S_12* - z* S_14 (Eq. app-eqn:Phi-out-final).

    Arguments
    ---------
    Phi_in : callable
        Target characteristic function, a function of the complex argument z.
    S_12, S_14 : complex
        Transduction and two-mode-squeezing amplitudes (first row of the scattering matrix).
    expect_NdN, expect_NN : complex
        Noise moments <N† N> and <N²> of the added-noise operator N.
    comm_NNd : float
        Noise commutator [N, N†] = 1 - |S_12|² + |S_14|².
    R : float
        Half-width of the (square) integration grid.
    N : int
        Number of grid points per axis.
    """

    # Create the integration grid in the complex plane
    axis = np.linspace(-R, R, N)
    X, Y = np.meshgrid(axis, axis)

    # Complex integration variable.
    z = X + 1j*Y

    # Rescaled argument for the output characteristic function.
    z_tilde = np.conj(S_12)*z - S_14*np.conj(z)

    # Noise factor in the output characteristic function.
    noise = np.exp(0.5*(
        z**2 * np.conj(expect_NN)
        + np.conj(z)**2 * expect_NN
        - np.abs(z)**2 * (2*expect_NdN + comm_NNd)
        ))

    # Integrand for the fidelity integral and grid spacing.
    integrand = Phi_in(z) * Phi_in(-z_tilde) * noise
    dz = axis[1] - axis[0]

    # Perform the double integral using the trapezoidal rule.
    return (np.trapezoid(np.trapezoid(integrand, dx=dz, axis=0), dx=dz) / np.pi).real
