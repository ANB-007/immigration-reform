# src/simulation/empirical_params.py
"""
Single source-of-truth for all empirical parameters used in the workforce simulation.
All real-world rates and proportions are defined here for easy modification.
CORRECTED FOR SPEC-8: Fixed annual conversions and enhanced wage differentiation.

Data sources (as of October 2025):
- BLS Employment Situation Report August 2025
- USCIS H-1B data FY 2024
- American Immigration Council reports 2024
- USCIS Employment-Based Green Card statistics 2024
- BLS Job Openings and Labor Turnover Survey (JOLTS) 2024
- Jennifer Hunt research on temporary worker job mobility
- USCIS H-1B Nationality Distribution FY 2024
- DOL H-1B Disclosure Data by Country of Birth 2024
- INA Section 203(b) Per-Country Limitation (7% rule)
"""

import math

# Technical requirements
PYTHON_MIN_VERSION = "3.10"
DEFAULT_SEED = 42

# Industry configuration (FROM SPEC-3)
INDUSTRY_NAME = "information_technology"  # Single industry for the simulation (string)

# Starting wages (FROM SPEC-3)
STARTING_WAGE = 95000.0          # USD per year (float) - starting wage for all workers

# Job-change probabilities (CORRECTED FOR SPEC-8)
# Interpreted as probability an individual changes jobs in a given year
JOB_CHANGE_PROB_PERM = 0.12     # 12% of permanent workers change jobs each year (float) - CORRECTED
TEMP_JOB_CHANGE_PENALTY = 0.20  # Temporary workers are 20% less likely to change jobs (Jennifer Hunt finding)
# Derived: JOB_CHANGE_PROB_TEMP = JOB_CHANGE_PROB_PERM * (1 - TEMP_JOB_CHANGE_PENALTY) = 0.096

# Wage jump when changing jobs (CORRECTED FOR SPEC-8 - ENHANCED DIFFERENTIATION)
# Interpreted as multiplicative factor applied to wage when an agent changes job
WAGE_JUMP_FACTOR_MEAN_PERM = 1.10    # Mean 10% wage boost for permanent workers (float) - CORRECTED
WAGE_JUMP_FACTOR_STD_PERM = 0.03     # Standard deviation for permanent wage jump (float)
WAGE_JUMP_FACTOR_MEAN_TEMP = 1.05    # Mean 5% wage boost for temporary workers (float) - CORRECTED
WAGE_JUMP_FACTOR_STD_TEMP = 0.03     # Standard deviation for temporary wage jump (float)

# Nationality distributions (FROM SPEC-4)
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

# Default nationality for permanent (domestic) workers (FROM SPEC-4)
PERMANENT_NATIONALITY = "United States"

# H-1B workforce proportions (based on current research)
# Current estimates suggest ~583,420 H-1B holders in 2019 (latest official count)
# With labor force of ~167 million in 2025, this gives us roughly 0.35%
H1B_SHARE = 0.0041                # ~0.41% of workforce are H-1B holders (conservative estimate)
PERMANENT_SHARE = 1 - H1B_SHARE   # Rest are permanent workers

# Annual entry rates (expressed as fractions of current total workforce)
# Based on BLS projections of 0.5% annual labor force growth
# New H-1B approvals: ~141,181 in FY 2024 out of ~167M workforce = ~0.0008
ANNUAL_PERMANENT_ENTRY_RATE = 0.0242   # Permanent workers entry rate (total growth minus H-1B)
ANNUAL_H1B_ENTRY_RATE = 0.0008         # H-1B entry rate based on 2024 approvals

# Green card transition parameters (CORRECTED FOR SPEC-8)
GREEN_CARD_CAP_ABS = 140_000           # Real-world annual employment-based green card cap
REAL_US_WORKFORCE_SIZE = 167_000_000   # Current US workforce size (August 2025)

# Per-country cap parameters (FROM SPEC-5)
# Based on INA Section 203(b) - Immigration and Nationality Act per-country limitation
PER_COUNTRY_CAP_SHARE = 0.07           # 7% per-country limit on employment-based green cards

# CORRECTED FOR SPEC-8: Fixed annual conversion parameters
CARRYOVER_FRACTION_STRATEGY = True     # Accumulate residual and grant extra when >=1
ENABLE_COUNTRY_CAP = False             # Toggle for per-country cap (can be overridden by CLI)

# Visualization options (FROM SPEC-6)
ENABLE_VISUALIZATION = True            # Default visualization toggle
SAVE_PLOTS = True                      # Whether to save plots automatically
OUTPUT_DIR = "output/"                 # Directory to store charts and figures
PLOT_DPI = 300                         # DPI for saved plots
PLOT_STYLE = "whitegrid"               # Seaborn style
PLOT_PALETTE = "muted"                 # Seaborn color palette

# FROM SPEC-8: Dashboard parameters
DASHBOARD_HOST = "127.0.0.1"          # Dashboard host address
DASHBOARD_PORT = 8050                  # Dashboard port
DASHBOARD_DEBUG = False                # Dashboard debug mode

# Simulation configuration
DEFAULT_YEARS = 30
TIMESTEP_YEARS = 1

# CORRECTED FOR SPEC-8: Derived parameters for clarity
JOB_CHANGE_PROB_TEMP = JOB_CHANGE_PROB_PERM * (1 - TEMP_JOB_CHANGE_PENALTY)  # 0.096

# Data sources and timestamps
DATA_SOURCES = {
    "bls_employment": "BLS Employment Situation August 2025",
    "bls_jolts": "BLS Job Openings and Labor Turnover Survey 2024",
    "uscis_h1b": "USCIS H-1B FY 2024 Reports", 
    "uscis_h1b_nationality": "USCIS H-1B Nationality Distribution FY 2024",
    "dol_h1b_disclosure": "DOL H-1B Disclosure Data by Country of Birth 2024",
    "uscis_green_cards": "USCIS Employment-Based Green Card FY 2024 Reports",
    "ina_per_country": "Immigration and Nationality Act Section 203(b) Per-Country Limitation",
    "labor_force_size": "167 million (Aug 2025)",
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
    "nationality_distribution_sum": (0.999, 1.001),  # Distribution must sum to ~1.0
    "per_country_cap_share": (0.01, 0.20)  # Per-country cap between 1% and 20%
}

# Validation function for nationality distribution (FROM SPEC-4)
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

# Validation for per-country cap parameters (FROM SPEC-5)
if not (0.01 <= PER_COUNTRY_CAP_SHARE <= 0.20):
    raise ValueError(f"PER_COUNTRY_CAP_SHARE must be between 1% and 20%, got {PER_COUNTRY_CAP_SHARE}")

# CORRECTED FOR SPEC-8: Validation for wage parameters
if WAGE_JUMP_FACTOR_MEAN_PERM <= WAGE_JUMP_FACTOR_MEAN_TEMP:
    raise ValueError(f"Permanent wage jump mean ({WAGE_JUMP_FACTOR_MEAN_PERM}) must be greater than temporary ({WAGE_JUMP_FACTOR_MEAN_TEMP})")

if JOB_CHANGE_PROB_PERM <= JOB_CHANGE_PROB_TEMP:
    raise ValueError(f"Permanent job change probability ({JOB_CHANGE_PROB_PERM}) must be greater than temporary ({JOB_CHANGE_PROB_TEMP})")

# Validate the default distribution
if not validate_nationality_distribution(TEMP_NATIONALITY_DISTRIBUTION):
    raise ValueError(f"TEMP_NATIONALITY_DISTRIBUTION does not sum to 1.0: {sum(TEMP_NATIONALITY_DISTRIBUTION.values())}")

# CORRECTED FOR SPEC-8: Helper functions for fixed conversion calculations
def calculate_annual_sim_cap(initial_workers: int) -> tuple[int, float]:
    """
    Calculate fixed annual simulation cap and residual fraction.
    
    Args:
        initial_workers: Initial workforce size
        
    Returns:
        Tuple of (annual_sim_cap, residual_fraction)
    """
    cap_proportion = GREEN_CARD_CAP_ABS / REAL_US_WORKFORCE_SIZE
    annual_sim_cap = math.floor(initial_workers * cap_proportion)
    residual_fraction = (initial_workers * cap_proportion) - annual_sim_cap
    
    return annual_sim_cap, residual_fraction

def calculate_per_country_cap(annual_sim_cap: int) -> tuple[int, float]:
    """
    Calculate per-country cap and residual fraction.
    
    Args:
        annual_sim_cap: Annual simulation conversion cap
        
    Returns:
        Tuple of (per_country_cap, residual_fraction)
    """
    per_country_cap = math.floor(annual_sim_cap * PER_COUNTRY_CAP_SHARE)
    residual_fraction = (annual_sim_cap * PER_COUNTRY_CAP_SHARE) - per_country_cap
    
    return per_country_cap, residual_fraction
