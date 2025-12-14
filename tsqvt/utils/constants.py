"""
Physical Constants
==================

Fundamental constants and Standard Model parameters.

All values from PDG 2024 unless otherwise noted.

References
----------
.. [1] PDG (2024). Review of Particle Physics.
"""

import numpy as np

# ==============================================================================
# Fundamental Constants (SI units)
# ==============================================================================

HBAR = 1.054571817e-34      # Reduced Planck constant [J·s]
C = 299792458               # Speed of light [m/s]
G = 6.67430e-11             # Gravitational constant [m³/(kg·s²)]
K_B = 1.380649e-23          # Boltzmann constant [J/K]
E_CHARGE = 1.602176634e-19  # Elementary charge [C]

# Planck units
M_PLANCK = np.sqrt(HBAR * C / G)                    # Planck mass [kg]
M_PLANCK_GEV = M_PLANCK * C**2 / E_CHARGE / 1e9     # ≈ 1.22e19 GeV
L_PLANCK = np.sqrt(HBAR * G / C**3)                 # Planck length [m]
T_PLANCK = np.sqrt(HBAR * G / C**5)                 # Planck time [s]

# ==============================================================================
# Standard Model Mass Scales (GeV)
# ==============================================================================

# Gauge boson masses
M_Z = 91.1876        # Z boson mass
M_W = 80.377         # W boson mass
M_HIGGS = 125.25     # Higgs boson mass

# Electroweak scale
V_HIGGS = 246.22     # Higgs VEV

# ==============================================================================
# Gauge Couplings (at M_Z)
# ==============================================================================

ALPHA_EM = 1 / 137.035999084    # Fine structure constant
ALPHA_EM_INV = 137.035999084    # 1/α
ALPHA_S = 0.1179                # Strong coupling at M_Z
SIN2_THETA_W = 0.23122          # Weak mixing angle

# Fermi constant
G_FERMI = 1.1663788e-5  # GeV⁻²

# ==============================================================================
# Fermion Masses (GeV)
# ==============================================================================

SM_MASSES = {
    # Charged leptons
    'e': 0.000511,
    'mu': 0.1057,
    'tau': 1.777,
    
    # Up-type quarks (MS-bar at μ = m)
    'u': 0.00216,
    'c': 1.27,
    't': 172.69,
    
    # Down-type quarks (MS-bar at μ = 2 GeV)
    'd': 0.00467,
    's': 0.0934,
    'b': 4.18,
    
    # Neutrinos (upper limits, eV converted to GeV)
    'nu_e': 1e-9,
    'nu_mu': 1e-9,
    'nu_tau': 1e-9,
}

# ==============================================================================
# CKM Matrix (Wolfenstein parametrization)
# ==============================================================================

# Wolfenstein parameters
LAMBDA_W = 0.22650      # sin(θ_C)
A_W = 0.790
RHO_BAR = 0.141
ETA_BAR = 0.357

# CKM magnitudes
SM_MIXINGS = {
    'V_ud': 0.97373,
    'V_us': 0.2243,
    'V_ub': 0.00382,
    'V_cd': 0.221,
    'V_cs': 0.975,
    'V_cb': 0.0410,
    'V_td': 0.0080,
    'V_ts': 0.0388,
    'V_tb': 1.013,
}

# PMNS matrix (neutrino mixing)
PMNS_ANGLES = {
    'theta_12': 33.44,   # degrees
    'theta_23': 49.2,
    'theta_13': 8.57,
    'delta_CP': 194,
}

# ==============================================================================
# Derived Quantities
# ==============================================================================

# Mass ratios
MASS_RATIOS = {
    'm_tau/m_mu': SM_MASSES['tau'] / SM_MASSES['mu'],  # ≈ 16.8
    'm_mu/m_e': SM_MASSES['mu'] / SM_MASSES['e'],      # ≈ 207
    'm_t/m_b': SM_MASSES['t'] / SM_MASSES['b'],        # ≈ 41.3
    'm_b/m_tau': SM_MASSES['b'] / SM_MASSES['tau'],    # ≈ 2.35
}

# Electroweak ratios
EW_RATIOS = {
    'M_W/M_Z': M_W / M_Z,                              # ≈ 0.8815
    'rho_0': M_W**2 / (M_Z**2 * (1 - SIN2_THETA_W)),  # ≈ 1
}

# ==============================================================================
# TSQVT Specific Parameters
# ==============================================================================

# Spectral manifold parameters
TSQVT_PARAMS = {
    'volume': 1.85e-61,        # m^4 (spectral manifold)
    'twist_angle': 0.198,      # rad
    'rho_ew': 0.742,           # Condensation at EW scale
    'rho_critical': 2/3,       # Critical condensation
    'n_generations': 3,
}

# Golden ratio (appears in generation structure)
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618

# ==============================================================================
# Unit Conversions
# ==============================================================================

GEV_TO_KG = E_CHARGE * 1e9 / C**2
KG_TO_GEV = 1 / GEV_TO_KG
GEV_TO_JOULE = E_CHARGE * 1e9
MEV_TO_GEV = 1e-3
TEV_TO_GEV = 1e3

# Natural units (ℏ = c = 1)
GEV_INV_TO_METER = HBAR * C / (E_CHARGE * 1e9)  # GeV⁻¹ to meters
GEV_INV_TO_SECOND = HBAR / (E_CHARGE * 1e9)     # GeV⁻¹ to seconds
