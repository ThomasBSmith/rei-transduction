"""
Functions for the analysis of rare-earth ion transduction & entanglement.
"""

from .transducer import (
    Transducer, 
    phase_correct, 
    aout_moments, 
    measured_efficiency
)

from .fidelity import (
    Phi_fock, 
    Phi_coherent, 
    Phi_gkp, 
    noise_factor, 
    Phi_out,
    integration_grid, 
    fidelity,
)

from .gkp_analytics import (
    gkp_envelope,
    gkp_channel_params,
    gkp_infidelity
)

from .gkp_numerics import (
    decoded_pauli,
    decoded_rho,
    raw_pauli,
    raw_rho,
    logical_fidelity,
)

from .entanglement import (
    spectral_covariance, 
    covariance, 
    log_negativity,
    filter_func, 
    filter_matrix,
)

__all__ = [
    "Transducer", "phase_correct", "aout_moments", "measured_efficiency",
    "Phi_fock", "Phi_coherent", "Phi_gkp", "noise_factor", "Phi_out", "integration_grid", "fidelity",
    "gkp_envelope", "gkp_channel_params", "gkp_infidelity",
    "decoded_pauli", "decoded_rho", "raw_pauli", "raw_rho", "logical_fidelity",
    "spectral_covariance", "covariance", "log_negativity", "filter_func", "filter_matrix",
]
