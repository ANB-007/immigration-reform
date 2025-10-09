# src/simulation/empirical_params.py
"""
Single source-of-truth for all empirical parameters used in the workforce simulation.
All real-world rates and proportions are defined here for easy modification.

Data sources (as of October 2025):
- BLS Employment Situation Report August 2025
- USCIS H-1B data FY 2024
- American Immigration Council reports 2024
- USCIS Employment-Based Green Card statistics 2024
"""

# Technical requirements
PYTHON_MIN_VERSION = "3.10"
DEFAULT_SEED = 42

# H-1B workforce proportions (based on current research)
# Current estimates suggest ~583,420 H-1B holders in 2019 (latest official count)
# With labor force of ~171 million in 2025, this gives us roughly 0.34%
# However, this may be conservative as numbers have grown since 2019
H1B_SHARE = 0.0041                # ~0.41% of workforce are H-1B holders (conservative estimate)
PERMANENT_SHARE = 1 - H1B_SHARE   # Rest are permanent workers

# Annual entry rates (expressed as fractions of current total workforce)
# Based on BLS projections of 0.5% annual labor force growth
# New H-1B approvals: ~141,181 in FY 2024 out of ~171M workforce = ~0.0008
ANNUAL_PERMANENT_ENTRY_RATE = 0.0242   # Permanent workers entry rate (total growth minus H-1B)
ANNUAL_H1B_ENTRY_RATE = 0.0008         # H-1B entry rate based on 2024 approvals

# Green card transition parameters (NEW FOR SPEC-2)
GREEN_CARD_CAP_ABS = 140_000           # Real-world annual employment-based green card cap
REAL_US_WORKFORCE_SIZE = 171_000_000   # Current US workforce size (August 2025)

# Simulation configuration
DEFAULT_YEARS = 30
TIMESTEP_YEARS = 1

# Data sources and timestamps
DATA_SOURCES = {
    "bls_employment": "BLS Employment Situation August 2025",
    "uscis_h1b": "USCIS H-1B FY 2024 Reports", 
    "uscis_green_cards": "USCIS Employment-Based Green Card FY 2024 Reports",
    "labor_force_size": "171 million (Aug 2025)",
    "h1b_approvals_2024": "141,181 new petitions",
    "green_card_cap_2024": "140,000 annual employment-based cap",
    "participation_rate": "62.3% (Aug 2025)"
}

# Validation ranges (for testing)
VALID_RANGES = {
    "h1b_share": (0.002, 0.008),       # 0.2% to 0.8% of workforce
    "annual_growth": (0.001, 0.05),    # 0.1% to 5% annual growth
    "simulation_years": (1, 50),       # 1 to 50 years max
    "green_card_proportion": (0.0001, 0.01)  # Green card cap proportion range
}
