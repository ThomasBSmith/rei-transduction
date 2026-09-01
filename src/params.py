"""
[Rochman2023] J. Rochman, T. Xie, J. G. Bartholomew, K. C. Schwab, A. Faraon,
  "Microwave-to-optical transduction with erbium ions coupled to planar photonic
  and superconducting resonators," Nat. Commun. 14, 1153 (2023).
  doi:10.1038/s41467-023-36799-0

[King2024] G. G. G. King, L. S. Trainor, J. J. Longdell, "Triply Resonant
  Microwave to Optical Conversion in Erbium-170 Doped Yttrium Orthosilicate,"
  arXiv:2409.14612 (2024).

[FG2019] X. Fernandez-Gonzalvo, S. P. Horvath, Y.-H. Chen, J. J. Longdell,
  "Cavity-enhanced Raman heterodyne spectroscopy in Er3+:Y2SiO5 for microwave to
  optical signal conversion," Phys. Rev. A 100, 033807 (2019). arXiv:1712.07735.
  Apparatus paper for the [King2024] resonators.

[Sahu2022] R. Sahu, W. Hease, A. Rueda, G. Arnold, L. Qiu, J. M. Fink,
  "Quantum-enabled operation of a microwave-optical interface,"
  Nat. Commun. 13, 1276 (2022). arXiv:2107.08303.
"""

# [Rochman2023, SI Table S4 & SI Note 8]
ROCHMAN_PARAMS = {
    'kappa_a' : 13.2,       # Optical loss rate [2π * GHz].
    'kappa_b' : 0.002,      # Microwave loss rate [2π * GHz].
    'omega_b' : 4.94,       # Microwave resonator frequency [2π * GHz].
    'eta_a' : 2.9/13.2,     # Optical out-coupling fraction.
    'eta_b' : 0.85/2.0,     # Microwave out-coupling fraction.
    'Delta_a' : 0.100,      # Optical drive detuning from the |0>-|2> transition [2π * GHz].
}

# [King2024, Sec. II]; out-coupling from [FG2019, p.5 & Table II].
KING_PARAMS = {
    'kappa_a' : 2.64/310,     # Optical loss rate (FSR / finesse) [2π * GHz].
    'kappa_b' : 0.002,        # Microwave loss rate [2π * GHz].
    'omega_b' : 5.8,          # Microwave resonator frequency [2π * GHz].
    'eta_a' : 8.0/9.7,        # Optical out-coupling fraction (over-coupled FP) [FG2019].
    'eta_b' : 1 - 0.717/2.0,  # Microwave out-coupling fraction: kappa_int/2π = 717 kHz & kappa_tot/2π = 2 MHz [FG2019].
    'Delta_a' : 0.170,        # Optical drive detuning from the |0>-|2> transition [2π * GHz] [FG2019].
}

# [Sahu2022, "Physics and implementation" section]; totals inferred as kappa_int / (1 - eta).
SAHU_PARAMS = {
    'kappa_a' : 10.8e-3/(1 - 0.58),  # Optical loss rate; kappa_int/2π = 10.8 MHz, eta_a = 0.58.
    'kappa_b' : 8.1e-3/(1 - 0.41),   # Microwave loss rate; kappa_int/2π = 8.1 MHz, eta_b = 0.41.
    'omega_b' : 8.795,               # Microwave resonator frequency [2π * GHz].
    'eta_a' : 0.58,                  # Optical out-coupling fraction (excludes Λ ≈ 0.78 mode-match).
    'eta_b' : 0.41,                  # Microwave out-coupling fraction.
}
