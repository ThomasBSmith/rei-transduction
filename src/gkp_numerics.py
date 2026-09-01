"""
Numerical GKP logical fidelity through the transduction channel (App. F3).

Built on the stabilizer-subsystem decomposition (Shaw et al., arXiv:2210.14919)
and the weighted-sum readout formulas of Matsos et al. (arXiv:2310.15546,
Eqs. S10-S12). Both treatments form the logical density matrix

    D(rho) = 1/2 (I + <X_m> X + <Y_m> Y + <Z_m> Z),

differing only in the Pauli expectations:

    decoded : S11 infinite-round ideal-decoding sums, equivalent to many rounds
              of perfect GKP error correction.
    raw     : bare single-displacement operators, no decoding (before QEC).

Characteristic-function convention matches src/fidelity.py: chi(b) = <D(b)>,
D(b) = exp(b a† - b* a), so Phi_out plugs straight in.
"""

import numpy as np

# Pauli matrices, computational basis |0>, |1>.
_I = np.array([[1, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def decoded_pauli(chi, trunc=3):
    """
    Decoded logical Pauli expectations (<X_m>, <Y_m>, <Z_m>) of a square-lattice
    GKP state, from the ideal-decoding sums of Matsos et al. Eq. S11:

        <X_m> = (1/pi)   Sum_n (-1)^n / (n + 1/2)  chi( sqrt(2 pi) (n + 1/2) )
        <Z_m> = (1/pi)   Sum_n (-1)^n / (n + 1/2)  chi( i sqrt(2 pi) (n + 1/2) )
        <Y_m> = (1/pi^2) Sum_{m,n} chi( sqrt(2 pi) [(m + 1/2) + i (n + 1/2)] )
                                   / [ (m + 1/2)(n + 1/2) ].

    The half-integer index runs |k + 1/2| <= trunc (Matsos et al. use trunc = 3).
    Convergence should be checked: the chi envelope decays slowly at large nbar.

    Arguments
    ---------
    chi : callable
        Characteristic function chi(b): the bare Phi_gkp or the transduced Phi_out.
    trunc : int
        Lattice truncation, keep |k + 1/2| <= trunc.

    Returns
    -------
    (x, y, z) : tuple of float
        Real decoded Pauli expectation values.
    """
    a = np.sqrt(2*np.pi)
    k = np.arange(-trunc, trunc) + 0.5           # half-integers, |k| < trunc
    w = (-1.0)**np.arange(-trunc, trunc) / k     # (-1)^n / (n + 1/2)

    x = (w * chi(a*k)).sum() / np.pi
    z = (w * chi(1j*a*k)).sum() / np.pi

    K1, K2 = np.meshgrid(k, k, indexing='ij')
    grid = a*(K1 + 1j*K2)
    y = (chi(grid) / (K1*K2)).sum() / np.pi**2

    return x.real, y.real, z.real


def decoded_rho(chi, trunc=3):
    """
    2x2 decoded logical density matrix D(rho) (Matsos et al. Eq. S10) from
    decoded_pauli(). Not projected onto the physical state set -- inspect the
    Bloch-vector norm to gauge leakage.
    """
    x, y, z = decoded_pauli(chi, trunc)
    return 0.5*(_I + x*_X + y*_Y + z*_Z)


def raw_pauli(chi):
    """
    Bare (undecoded) logical Pauli expectations of a square-lattice GKP state.
    In the stabilizer-subsystem decomposition (Shaw et al. Eq. 25) the logical
    operators are single displacements,

        X_bar = D(b),   Z_bar = D(i b),   Y_bar = D(b + i b),   b = sqrt(pi/2),

    so each expectation is one point evaluation of chi (no decoding sum). The
    gap to decoded_pauli() is the benefit of ideal GKP error correction.

    Arguments
    ---------
    chi : callable
        Characteristic function chi(b).

    Returns
    -------
    (x, y, z) : tuple of float
        Real bare Pauli expectation values.
    """
    b = np.sqrt(np.pi/2)
    x = chi(np.array(b))
    z = chi(np.array(1j*b))
    y = chi(np.array(b + 1j*b))   # Y_bar = i X_bar Z_bar = D(b + i b)
    return np.real(x), np.real(y), np.real(z)


def raw_rho(chi):
    """
    2x2 bare logical density matrix D(rho) from raw_pauli() -- the same scaffold
    as decoded_rho() with the undecoded single-displacement operators.
    """
    x, y, z = raw_pauli(chi)
    return 0.5*(_I + x*_X + y*_Y + z*_Z)


def logical_fidelity(rho, sigma):
    """
    Uhlmann fidelity of two 2x2 density matrices, via the closed qubit form
    (Jozsa 1994),

        F = Tr(rho sigma) + 2 sqrt( det(rho) det(sigma) ).

    Same convention as the state fidelity in src/fidelity.py (F -> |<psi|phi>|^2
    for pure states). det is clipped at 0 to absorb the small negative
    eigenvalues that the unprojected decoded_rho can carry.
    """
    fid = np.trace(rho @ sigma).real
    fid += 2*np.sqrt(max(np.linalg.det(rho).real, 0.0)
                     * max(np.linalg.det(sigma).real, 0.0))
    return fid
