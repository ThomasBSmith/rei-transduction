"""
Analytic (closed-form) GKP average gate fidelity through the transduction
channel (App. F2; Shaw et al., arXiv:2210.14919).
"""

import numpy as np
from scipy.special import erfc


def gkp_envelope(nbar):
    """
    Gaussian-envelope parameter of a finite-energy square-lattice GKP state,

        Delta = 1/sqrt(2 nbar + 1).

    Arguments
    ---------
    nbar : float
        Mean photon number of the GKP state.
    """
    return 1 / np.sqrt(2*nbar + 1)


def gkp_channel_params(kappa_a, kappa_b, omega_b, z, eta_a, eta_b):
    """
    Impedance-matched GKP channel parameters at the central frequency (App. F2):
    the idealised efficiency eta, the finite-squeezing correction delta, and the
    phase-sensitive Gaussian noise variances sigma_G+^2, sigma_G-^2. Loss rates
    enter as ratios to omega_b.

    Arguments
    ---------
    kappa_a, kappa_b : float
        Optical and microwave linewidths.
    omega_b : float
        Microwave frequency.
    z : float
        Parasitic asymmetry |g_bp/g_bs|.
    eta_a, eta_b : float
        External coupling fractions kappa^ext / kappa.
    """
    ka, kb = kappa_a/omega_b, kappa_b/omega_b
    ka_ext = eta_a * ka

    eta   = eta_a * eta_b
    delta = z**2 * kb**2 / 16
    base  = z**2/32 * (2*ka_ext**2 + eta*kb**2)
    cross = (z*ka_ext/4) * (ka_ext/ka - 1)
    return eta, delta, base + cross, base - cross


def gkp_infidelity(nbar, eta, delta, sigmaG_sq_p, sigmaG_sq_m):
    """
    Average gate infidelity 1 - Fbar of a square-lattice GKP state transduced
    through the channel (App. F2; Shaw et al. logical-error formula). Sums the
    squeezed and anti-squeezed quadrature contributions.

    Arguments
    ---------
    nbar : float
        Mean photon number of the GKP state.
    eta : float
        Idealised transduction efficiency.
    delta : float
        Finite-squeezing correction to the efficiency.
    sigmaG_sq_p, sigmaG_sq_m : float
        Gaussian noise variances sigma_G+^2, sigma_G-^2.
    """
    Delta = gkp_envelope(nbar)
    total = 0.0
    for sign, sigmaG_sq in ((+1, sigmaG_sq_p), (-1, sigmaG_sq_m)):
        gamma = 1 - eta*(1 + sign*delta)
        root = np.sqrt(1 - gamma)
        sigma_sq = (root*np.tanh(Delta**2/2) + (1 - eta)/2
                    + (1 - root)**2/(2*np.tanh(Delta**2)) + sigmaG_sq)
        total += (2/3) * erfc(np.sqrt(np.pi / (8*sigma_sq)))
    return total
