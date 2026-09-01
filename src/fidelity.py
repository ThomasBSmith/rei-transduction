"""
State fidelities in phase space via characteristic functions (App. F1).
"""

import numpy as np
from scipy.special import eval_laguerre

from .gkp_analytics import gkp_envelope


def Phi_fock(z, n):
    """
    Characteristic function of the number state |n>,

        Phi(z) = e^{-|z|^2/2} L_n(|z|^2),

    with L_n the n-th Laguerre polynomial.

    Arguments
    ---------
    z : complex or ndarray
        Argument of the characteristic function.
    n : int
        Fock state number.
    """
    return np.exp(-np.abs(z)**2/2) * eval_laguerre(n, np.abs(z)**2)


def Phi_coherent(z, alpha):
    """
    Characteristic function of the coherent state |alpha>,

        Phi(z) = e^{-|z|^2/2} e^{z alpha* - z* alpha}.

    Arguments
    ---------
    z : complex or ndarray
        Argument of the characteristic function.
    alpha : complex
        Coherent state amplitude.
    """
    return np.exp(-np.abs(z)**2/2) * np.exp(z*np.conj(alpha) - np.conj(z)*alpha)


def Phi_gkp(z, nbar, mu=0, tol=1e-6):
    """
    Characteristic function of the finite-energy square-lattice GKP state
    |mu_gkp>, from Albert et al. (arXiv:1708.05010v3) Eq. (D11) with mu = nu:

        Phi(z) = (1/N) Sum_{n1,n2} (-1)^{(n1 + mu) n2}
                 exp(-|z - L|^2 / 2 Delta^2) exp(-Delta^2 |z + L|^2 / 8),

    lattice L = sqrt(pi/2) (2 n1 + i n2), envelope Delta = 1/sqrt(2 nbar + 1)
    (gkp_envelope), normalisation N = Phi(0). Same displacement convention
    D(z) = exp(z a† - z* a) as Phi_fock / Phi_coherent. Real and even in z.

    Arguments
    ---------
    z : complex or ndarray
        Argument of the characteristic function.
    nbar : float
        Mean photon number, sets the envelope Delta.
    mu : int
        Logical Z-basis state, 0 or 1.
    tol : float
        Smallest lattice-term amplitude kept in the sum.
    """
    Delta = gkp_envelope(nbar)
    z = np.asarray(z, dtype=complex)
    a = np.sqrt(np.pi / 2)   # L = a (2 n1 + i n2)

    # Truncate where the term amplitude exp(-Delta^2 |L|^2 / 2) drops below tol.
    Lam_max = np.sqrt(2 * np.log(1/tol)) / Delta
    n1_max = int(np.ceil(Lam_max / (2*a)))
    n2_max = int(np.ceil(Lam_max / a))

    def lattice_sum(w):
        out = np.zeros(np.shape(w), dtype=complex)
        for n1 in range(-n1_max, n1_max + 1):
            for n2 in range(-n2_max, n2_max + 1):
                Lam = a * (2*n1 + 1j*n2)
                if np.exp(-Delta**2 * np.abs(Lam)**2 / 2) < tol:
                    continue
                out += (-1.0)**((n1 + mu)*n2) * (
                    np.exp(-np.abs(w - Lam)**2 / (2*Delta**2))
                    * np.exp(-Delta**2 * np.abs(w + Lam)**2 / 8)
                    )
        return out

    return (lattice_sum(z) / lattice_sum(np.array(0.0))).real


def integration_grid(kind, param, tol=1e-6, pts_per_feature=8):
    """
    Integration grid (R, N) for fidelity(), sized to the target characteristic
    function so the defaults need not be hand-tuned per state.

    R is the half-width where |Phi_in| has decayed to tol. The spacing puts
    pts_per_feature points across the fastest oscillation of Phi_in, floored so
    the O(1)-width Gaussian envelope is always resolved.

        coherent : envelope width 1; phase winds with wavelength pi/sqrt(nbar).
        fock     : e^{-s/2} s^n (s = |z|^2) peaks at s = 2n; L_n zeros spaced
                   pi/sqrt(n) near the origin.
        gkp      : comb envelope exp(-Delta^2 |z|^2 / 8), peak width Delta;
                   1 - F is a small residual after +/- peak cancellation, so the
                   spacing over-resolves (~Delta/4). Cost ~ N^2 ~ nbar^2.

    Arguments
    ---------
    kind : {'coherent', 'fock', 'gkp'}
        Target state.
    param : float or int
        Mean photon number nbar ('coherent', 'gkp') or Fock number n ('fock').
    tol : float
        Target decay of |Phi_in| at the grid edge.
    pts_per_feature : int
        Samples across the fastest oscillation.
    """
    ln = np.log(1/tol)

    if kind == 'coherent':
        R = np.sqrt(2*ln) + 1.0
        feature = np.pi/np.sqrt(param) if param > 0 else np.inf
    elif kind == 'fock':
        R = np.sqrt(2*param + 8*np.sqrt(param + 1) + 2*ln) + 1.0
        feature = np.pi/np.sqrt(param) if param > 0 else np.inf
    elif kind == 'gkp':
        Delta = gkp_envelope(param)
        R = np.sqrt(2*ln)/Delta + 3*Delta
        feature = 2*Delta
    else:
        raise ValueError(f"unknown kind {kind!r}")

    h = min(feature/pts_per_feature, 0.5)
    N = 2*int(np.ceil(R/h)) + 1
    return R, N


def noise_factor(z, expect_NdN, expect_NN, comm_NNd):
    """
    Gaussian added-noise factor the channel applies to Phi_in (App. F1),

        exp{ (1/2) [ z^2 <N†^2> + z*^2 <N^2> - |z|^2 (2 <N†N> + [N,N†]) ] },

    from the added-noise moments of aout_moments. Even in z.

    Arguments
    ---------
    z : complex or ndarray
        Argument of the characteristic function.
    expect_NdN, expect_NN : complex
        Noise moments <N†N> and <N^2>.
    comm_NNd : float
        Noise commutator [N, N†] = 1 - |S_12|^2 + |S_14|^2.
    """
    return np.exp(0.5*(
        z**2 * np.conj(expect_NN)
        + np.conj(z)**2 * expect_NN
        - np.abs(z)**2 * (2*expect_NdN + comm_NNd)
        ))


def Phi_out(z, Phi_in, S_12, S_14, expect_NdN, expect_NN, comm_NNd):
    """
    Transduced characteristic function (App. F1, Eq. app-eqn:Phi-out-final),

        Phi_out(z) = Phi_in(S_12* z - S_14 z*) * noise_factor(z).

    The channel is (S_12, S_14, <N†N>, <N^2>, [N,N†]) from aout_moments.

    Arguments
    ---------
    z : complex or ndarray
        Argument of the characteristic function.
    Phi_in : callable
        Input characteristic function.
    S_12, S_14 : complex
        Transduction and two-mode-squeezing amplitudes (first row of S).
    expect_NdN, expect_NN : complex
        Noise moments <N†N> and <N^2>.
    comm_NNd : float
        Noise commutator [N, N†].
    """
    z_tilde = np.conj(S_12)*z - S_14*np.conj(z)
    return Phi_in(z_tilde) * noise_factor(z, expect_NdN, expect_NN, comm_NNd)


def fidelity(Phi_in, S_12, S_14, expect_NdN, expect_NN, comm_NNd, R=10.0, N=501):
    """
    Transduction fidelity (App. F1, Eq. app-eqn:fidelity),

        F = (1/pi) integral d^2z  Phi_in(z) Phi_out(-z),

    on a square grid of half-width R with N points per axis (integration_grid).

    Arguments
    ---------
    Phi_in : callable
        Target characteristic function.
    S_12, S_14 : complex
        Transduction and two-mode-squeezing amplitudes (first row of S).
    expect_NdN, expect_NN : complex
        Noise moments <N†N> and <N^2>.
    comm_NNd : float
        Noise commutator [N, N†].
    R : float
        Half-width of the integration grid.
    N : int
        Grid points per axis.
    """

    # Integration grid in the complex plane.
    axis = np.linspace(-R, R, N)
    X, Y = np.meshgrid(axis, axis)
    z = X + 1j*Y

    # Fidelity integrand and grid spacing.
    integrand = Phi_in(z) * Phi_out(-z, Phi_in, S_12, S_14, expect_NdN, expect_NN, comm_NNd)
    dz = axis[1] - axis[0]

    # Double integral by the trapezoidal rule.
    return (np.trapezoid(np.trapezoid(integrand, dx=dz, axis=0), dx=dz) / np.pi).real
