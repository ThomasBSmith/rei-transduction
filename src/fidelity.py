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


def Phi_coherent(z, beta):
    """
    Characteristic function of the coherent state |beta>.

    Arguments
    ---------
    z : complex
        Complex argument of the characteristic function.
    beta : complex
        Coherent state amplitude.
    """
    return np.exp(-np.abs(z)**2/2) * np.exp(z*np.conj(beta) - np.conj(z)*beta)


def fidelity(Phi_in, mu, nu, NdN, NN, comm, R=10.0, N=501):
    """
    Transduction fidelity F = (1/π) ∫ d²z Φ_in(z) Φ_out(-z), evaluated on a square
    grid in the complex plane. The output characteristic function is

        Φ_out(-z) = Φ_in(-z̃) exp{ ½[ z² <N†²> + z*² <N²> - |z|²(2<N†N> + [N,N†]) ] },

    with the rescaled argument z̃ = μ * z - ν * conj(z).

    Arguments
    ---------
    Phi_in : callable
        Target characteristic function, a function of the complex argument z.
    mu, nu : complex
        Signal transfer amplitudes S_12, S_14.
    NdN, NN : complex
        Noise moments <N†N> and <N²>.
    comm : float
        Noise commutator [N, N†].
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

    # Shifted argument for the output characteristic function.
    z_tilde = np.conj(mu)*z - nu*np.conj(z)

    # Noise factor in the output characteristic function.
    noise = np.exp(0.5*(z**2 * np.conj(NN) + np.conj(z)**2 * NN - np.abs(z)**2 * (2*NdN + comm)))

    # Integrand for the fidelity integral and grid spacing.
    integrand = Phi_in(z) * Phi_in(-z_tilde) * noise
    dz = axis[1] - axis[0]

    # Perform the double integral using the trapezoidal rule.
    return (np.trapezoid(np.trapezoid(integrand, dx=dz, axis=0), dx=dz) / np.pi).real
