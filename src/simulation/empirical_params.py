# src/simulation/empirical_params.py
"""
Single source-of-truth for all empirical parameters used in the workforce simulation.
All real-world rates and proportions are defined here for easy modification.

Data sources (as of October 2025):
- BLS Employment Situation Report August 2025
- USCIS H-1B data FY 2024
- American Immigration Council reports 2024
- USCIS Employment-Based Green Card statistics 2024
- BLS Job Openings and Labor Turnover Survey (JOLTS) 2024
- Jennifer Hunt research on temporary worker job mobility
- USCIS H-1B Nationality Distribution FY 2024
- DOL H-1B Disclosure Data by Country of Birth 2024
"""

# Technical requirements
PYTHON_MIN_VERSION = "3.10"
DEFAULT_SEED = 42

# Industry configuration (FROM SPEC-3)
INDUSTRY_NAME = "information_technology"  # Single industry for the simulation (string)

# Starting wages (FROM SPEC-3)
STARTING_WAGE = 95000.0          # USD per year (float) - starting wage for all workers

# Job-change probabilities (FROM SPEC-3)
# Interpreted as probability an individual changes jobs in a given year
JOB_CHANGE_PROB_PERM = 0.10     # 10% of permanent workers change jobs each year (float)
TEMP_JOB_CHANGE_PENALTY = 0.20  # Temporary workers are 20% less likely to change jobs (Jennifer Hunt finding)
# Derived: JOB_CHANGE_PROB_TEMP = JOB_CHANGE_PROB_PERM * (1 - TEMP_JOB_CHANGE_PENALTY)

# Wage jump when changing jobs (FROM SPEC-3)
# Interpreted as multiplicative factor applied to wage when an agent changes job
WAGE_JUMP_FACTOR_MEAN = 1.08    # Mean 8% wage boost on job change (float)
WAGE_JUMP_FACTOR_STD = 0.02     # Standard deviation for wage jump stochasticity (float)

# Nationality distributions (NEW FOR SPEC-4)
# Real-world shares based on USCIS H-1B data and DOL disclosure data (FY 2024 figures)
TEMP_NATIONALITY_DISTRIBUTION = {
    "India": 0.70,
    "China": 0.10,
    "Canada": 0.04,
    "South Korea": 0.03,
    "Philippines": 0.02,
    "United Kingdom": 0.02,
    "Mexico": 0.02,
    "Brazil": 0.01,
    "Germany": 0.01,
    "Other": 0.05,
}

# Default nationality for permanent (domestic) workers (NEW FOR SPEC-4)
PERMANENT_NATIONALITY = "United States"

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

# Green card transition parameters (FROM SPEC-2)
GREEN_CARD_CAP_ABS = 140_000           # Real-world annual employment-based green card cap
REAL_US_WORKFORCE_SIZE = 171_000_000   # Current US workforce size (August 2025)

# Simulation configuration
DEFAULT_YEARS = 30
TIMESTEP_YEARS = 1

# Data sources and timestamps
DATA_SOURCES = {
    "bls_employment": "BLS Employment Situation August 2025",
    "bls_jolts": "BLS Job Openings and Labor Turnover Survey 2024",
    "uscis_h1b": "USCIS H-1B FY 2024 Reports", 
    "uscis_h1b_nationality": "USCIS H-1B Nationality Distribution FY 2024",
    "dol_h1b_disclosure": "DOL H-1B Disclosure Data by Country of Birth 2024",
    "uscis_green_cards": "USCIS Employment-Based Green Card FY 2024 Reports",
    "labor_force_size": "171 million (Aug 2025)",
    "h1b_approvals_2024": "141,181 new petitions",
    "green_card_cap_2024": "140,000 annual employment-based cap",
    "participation_rate": "62.3% (Aug 2025)",
    "hunt_research": "Jennifer Hunt - Temporary worker job mobility research",
    "wage_data": "BLS Occupational Employment Statistics IT sector 2024"
}

# Validation ranges (for testing)
VALID_RANGES = {
    "h1b_share": (0.002, 0.008),       # 0.2% to 0.8% of workforce
    "annual_growth": (0.001, 0.05),    # 0.1% to 5% annual growth
    "simulation_years": (1, 50),       # 1 to 50 years max
    "green_card_proportion": (0.0001, 0.01),  # Green card cap proportion range
    "starting_wage": (50000, 200000),  # Reasonable wage range for IT sector
    "job_change_prob": (0.01, 0.30),   # 1% to 30% annual job change probability
    "wage_jump_factor": (1.01, 1.25),  # 1% to 25% wage jump on job change
    "nationality_distribution_sum": (0.999, 1.001)  # Distribution must sum to ~1.0
}

# Validation function for nationality distribution (NEW FOR SPEC-4)
def validate_nationality_distribution(distribution: dict, tolerance: float = 1e-6) -> bool:
    """
    Validate that nationality distribution sums to 1.0 within tolerance.
    
    Args:
        distribution: Dictionary mapping nationality to proportion
        tolerance: Acceptable deviation from 1.0
        
    Returns:
        True if distribution is valid
    """
    total = sum(distribution.values())
    return abs(total - 1.0) <= tolerance

# Validate the default distribution
if not validate_nationality_distribution(TEMP_NATIONALITY_DISTRIBUTION):
    raise ValueError(f"TEMP_NATIONALITY_DISTRIBUTION does not sum to 1.0: {sum(TEMP_NATIONALITY_DISTRIBUTION.values())}")
