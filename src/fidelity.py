"""
Tools for calculating state fidelities in phase-space using characteristic functions (App. F1).
"""

import numpy as np
from scipy.special import eval_laguerre

from .gkp import gkp_envelope


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


def Phi_gkp(z, nbar, mu=0, tol=1e-6):
    """
    Characteristic function of the finite-energy square-lattice GKP logical
    state |mu_gkps>, from Albert et al. (arXiv:1708.05010v3) Eq. (D11) with
    delta_mu = 0 (diagonal, mu = nu):

        Φ_gkp(z) = (1/𝒩) Σ_{n1,n2} (-1)^{(n1+mu) n2}
                   exp(-|z - Λ|² / 2Δ²) exp(-Δ² |z + Λ|² / 8),

    with lattice points Λ = sqrt(π/2) (2 n1 + i n2), envelope
    Δ = 1/sqrt(2 nbar + 1) (gkp_envelope), and normalisation 𝒩 = Φ_gkp(0)
    enforcing Tr ρ = 1. Same displacement convention D(z) = exp(z a† - z* a)
    as Phi_fock / Phi_coherent, so this plugs straight into fidelity().

    The lattice sum is real and even in z. Each term's amplitude is bounded by
    the energy envelope at its own peak, exp(-Δ² |Λ|² / 2), so the sum is
    truncated at |Λ|_max = sqrt(2 ln(1/tol)) / Δ (grid-independent). Note the
    envelope decays only as exp(-Δ² |z|² / 8), so fidelity() needs a wider
    window than the Fock/coherent default, e.g. R = 14 for nbar up to ~5.

    Arguments
    ---------
    z : complex or ndarray
        Complex argument of the characteristic function.
    nbar : float
        Mean photon number of the GKP state (sets the envelope Δ).
    mu : int
        Logical Z-basis state, 0 or 1.
    tol : float
        Smallest lattice-term amplitude retained in the sum.
    """
    Delta = gkp_envelope(nbar)
    z = np.asarray(z, dtype=complex)
    a = np.sqrt(np.pi / 2)   # Λ = a (2 n1 + i n2)

    # Truncate where the term amplitude exp(-Δ²|Λ|²/2) drops below tol.
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
    Integration grid (R, N) for fidelity(), sized to the characteristic
    function Phi_<kind> of the target state so the (R, N) defaults need not be
    hand-tuned per state.

    R is the half-width at which |Phi_in| has fallen to tol; the spacing h puts
    pts_per_feature samples across the fastest feature of Phi_in, floored at 0.5
    to always resolve the O(1)-width Gaussian envelope. All three scalings were
    checked against a >=10x finer reference grid (see below).

    kind : {'coherent', 'fock', 'gkp'}
    param : float or int
        Mean photon number n̄ for 'coherent' and 'gkp'; Fock number n for 'fock'.

    coherent : Phi = e^{-|z|²/2} e^{2i Im(z α*)}, α = sqrt(n̄).
        Gaussian envelope of width 1, R = sqrt(2 ln 1/tol) + 1; the phase winds
        with wavelength π / sqrt(n̄). Exact to 5 sig figs for n̄ <= 15.
    fock : Phi = e^{-|z|²/2} L_n(|z|²).
        e^{-s/2} s^n / n! (s = |z|²) peaks at s = 2n with width ~2 sqrt(n), so
        R = sqrt(2n + 8 sqrt(n+1) + 2 ln 1/tol) + 1. The zeros of L_n(|z|²) have
        density (1/π) sqrt(4n - |z|²) in |z|, densest at the origin, so the
        tightest zero-pair wavelength is π / sqrt(n). Exact for n <= 15.
    gkp : Phi is the alternating-sign lattice comb of Phi_gkp, envelope
        exp(-Δ²|z|²/8), Δ = 1/sqrt(2 n̄ + 1).
        R = sqrt(2 ln 1/tol) / Δ + 3Δ. The spacing must be ~Δ/4, well below the
        comb peak width Δ: 1 - F is a small residual left after near-total
        cancellation between +/- peaks, so the peaks need over-resolving.
        Verified to 5 sig figs for n̄ <= 5. Cost ~ N² ~ 1/Δ⁴ ~ n̄² -- low n̄ only.
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
    Gaussian added-noise factor of the transduced characteristic function,

        exp{ ½[ z² <N†²> + z*² <N²> - |z|²(2<N†N> + [N,N†]) ] },

    from the moments of the added-noise operator N (aout_moments). This is the
    multiplicative penalty the channel applies to Φ_in, on top of the signal
    rescaling z -> S_12* z - S_14 z*. Even in z.

    Arguments
    ---------
    z : complex or ndarray
        Complex argument of the characteristic function.
    expect_NdN, expect_NN : complex
        Noise moments <N† N> and <N²> of the added-noise operator N.
    comm_NNd : float
        Noise commutator [N, N†] = 1 - |S_12|² + |S_14|².
    """
    return np.exp(0.5*(
        z**2 * np.conj(expect_NN)
        + np.conj(z)**2 * expect_NN
        - np.abs(z)**2 * (2*expect_NdN + comm_NNd)
        ))


def Phi_out(z, Phi_in, S_12, S_14, expect_NdN, expect_NN, comm_NNd):
    """
    Output (transduced) characteristic function (App. F1, Eq. app-eqn:Phi-out-final),

        Φ_out(z) = Φ_in(S_12* z - S_14 z*) · noise_factor(z),

    for an input state with characteristic function Phi_in sent through the
    channel described by (S_12, S_14, <N†N>, <N²>, [N,N†]) from aout_moments.
    The fidelity integrand is Φ_in(z) Φ_out(-z).

    Arguments
    ---------
    z : complex or ndarray
        Complex argument of the characteristic function.
    Phi_in : callable
        Input characteristic function, a function of the complex argument z.
    S_12, S_14 : complex
        Transduction and two-mode-squeezing amplitudes (first row of S).
    expect_NdN, expect_NN : complex
        Noise moments <N† N> and <N²> of the added-noise operator N.
    comm_NNd : float
        Noise commutator [N, N†] = 1 - |S_12|² + |S_14|².
    """
    z_tilde = np.conj(S_12)*z - S_14*np.conj(z)
    return Phi_in(z_tilde) * noise_factor(z, expect_NdN, expect_NN, comm_NNd)


def fidelity(Phi_in, S_12, S_14, expect_NdN, expect_NN, comm_NNd, R=10.0, N=501):
    """
    Transduction fidelity F = (1/π) ∫ d²z Φ_in(z) Φ_out(-z) (App. F1,
    Eq. app-eqn:fidelity), evaluated on a square grid in the complex plane,
    with Φ_out built by Phi_out().

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

    # Integration grid in the complex plane.
    axis = np.linspace(-R, R, N)
    X, Y = np.meshgrid(axis, axis)
    z = X + 1j*Y

    # Fidelity integrand and grid spacing.
    integrand = Phi_in(z) * Phi_out(-z, Phi_in, S_12, S_14, expect_NdN, expect_NN, comm_NNd)
    dz = axis[1] - axis[0]

    # Double integral by the trapezoidal rule.
    return (np.trapezoid(np.trapezoid(integrand, dx=dz, axis=0), dx=dz) / np.pi).real
