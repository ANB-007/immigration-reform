# src/simulation/utils.py
"""
Utility functions for the workforce simulation.
SPEC-10: Streamlined validation, fixed function signatures, removed redundant calculations.
"""

import json
import csv
import logging
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd

from .models import SimulationState, Worker, SimulationConfig, BacklogAnalysis
from .empirical_params import calculate_annual_conversion_cap, GREEN_CARD_CAP_ABS, REAL_US_WORKFORCE_SIZE

logger = logging.getLogger(__name__)

def compute_annual_slots_flat(initial_workers: int,
                              green_card_cap_abs: int = GREEN_CARD_CAP_ABS,
                              real_us_workforce_size: int = REAL_US_WORKFORCE_SIZE) -> int:
    """
    Compute flat annual conversion slots.
    SPEC-10: Delegates to empirical_params for consistency.
    """
    return calculate_annual_conversion_cap(initial_workers)

def calculate_compound_growth(initial_value: float, rate: float, periods: int) -> int:
    """
    Calculate compound growth over multiple periods.
    SPEC-10: Returns integer as expected by tests.
    """
    if periods <= 0:
        return int(round(initial_value))
    
    result = initial_value * ((1 + rate) ** periods)
    return int(round(result))

def save_backlog_analysis(backlog_analysis: BacklogAnalysis, filepath: str) -> None:
    """Save backlog analysis to CSV file."""
    try:
        backlog_analysis.save_csv(filepath)
        logger.info(f"Saved backlog analysis to {filepath}")
    except Exception as e:
        logger.error(f"Error saving backlog analysis: {e}")
        raise

def load_backlog_analysis(filepath: str) -> BacklogAnalysis:
    """
    Load backlog analysis from CSV file.
    SPEC-10: Returns BacklogAnalysis object (not DataFrame) with proper error handling.
    """
    try:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return BacklogAnalysis.load_csv(filepath)
        
    except Exception as e:
        logger.error(f"Error loading backlog analysis: {e}")
        raise

def save_simulation_results(states: List[SimulationState], 
                          output_path: str,
                          metadata: Optional[Dict[str, Any]] = None,
                          include_nationality_columns: Optional[bool] = None) -> None:
    """
    Save simulation results to CSV file.
    SPEC-10: Streamlined with consistent parameter handling.
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            # Write metadata as comments if provided
            if metadata:
                for key, value in metadata.items():
                    csvfile.write(f"# {key}: {value}\n")
            
            writer = csv.writer(csvfile)
            
            # Header row
            header = [
                'year', 'total_workers', 'permanent_workers', 'temporary_workers',
                'new_permanent', 'new_temporary', 'converted_temps',
                'avg_wage_total', 'avg_wage_permanent', 'avg_wage_temporary',
                'total_wage_bill', 'h1b_share', 'permanent_share',
                'cumulative_conversions', 'annual_conversion_cap'
            ]
            writer.writerow(header)
            
            # Data rows
            for state in states:
                writer.writerow([
                    state.year,
                    state.total_workers,
                    state.permanent_workers,
                    state.temporary_workers,
                    state.new_permanent,
                    state.new_temporary,
                    state.converted_temps,
                    round(state.avg_wage_total, 2),
                    round(state.avg_wage_permanent, 2),
                    round(state.avg_wage_temporary, 2),
                    round(state.total_wage_bill, 2),
                    round(state.h1b_share, 4),
                    round(state.permanent_share, 4),
                    state.cumulative_conversions,
                    state.annual_conversion_cap
                ])
        
        logger.info(f"Saved simulation results to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving simulation results: {e}")
        raise

def load_simulation_results(input_path: str) -> List[SimulationState]:
    """
    Load simulation results from CSV file.
    SPEC-10: Returns SimulationState objects with computed worker lists.
    """
    states = []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as csvfile:
            # Skip comment lines
            lines = []
            for line in csvfile:
                if not line.startswith('#'):
                    lines.append(line)
            
            # Parse CSV
            reader = csv.DictReader(lines)
            for row in reader:
                # Create empty workers list - would need to reconstruct from separate file
                state = SimulationState(
                    year=int(row['year']),
                    workers=[],  # Empty workers list from CSV
                    annual_conversion_cap=int(row.get('annual_conversion_cap', 0)),
                    new_permanent=int(row['new_permanent']),
                    new_temporary=int(row['new_temporary']),
                    converted_temps=int(row['converted_temps']),
                    cumulative_conversions=int(row.get('cumulative_conversions', 0))
                )
                states.append(state)
        
        logger.info(f"Loaded {len(states)} simulation states from {input_path}")
        return states
        
    except Exception as e:
        logger.error(f"Error loading simulation results: {e}")
        raise

def validate_configuration(config: SimulationConfig) -> List[str]:
    """
    Validate simulation configuration.
    SPEC-10: Returns list of error messages, only meaningful validations.
    """
    errors = []
    
    try:
        # Only validate constraints that can be violated by user input
        if config.initial_workers <= 0:
            errors.append("initial_workers must be positive")
        
        if config.years <= 0:
            errors.append("years must be positive")
        
        # Reasonable bounds checking
        if config.years > 50:
            errors.append("years should not exceed 50 for reasonable execution time")
        
        if config.initial_workers > 1000000:
            errors.append("initial_workers should not exceed 1,000,000 for reasonable execution time")
        
        # Backlog analysis requirements
        if config.compare_backlogs and config.initial_workers < 5000:
            errors.append("compare_backlogs requires at least 5000 initial workers for meaningful analysis")
        
        return errors
        
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return [str(e)]

def format_currency(amount: float) -> str:
    """Format currency amount for display."""
    return f"${amount:,.2f}"

def format_number(number: int) -> str:
    """Format number with thousands separators."""
    return f"{number:,}"

def format_percentage(value: float, precision: Optional[int] = None) -> str:
    """
    Format percentage for display.
    SPEC-10: Supports precision parameter for test compatibility.
    """
    if precision is None:
        precision = 2
    
    format_str = f"{{:.{precision}%}}"
    return format_str.format(value)

def analyze_conversion_queue_efficiency(states: List[SimulationState]) -> Dict[str, Any]:
    """
    Analyze conversion queue efficiency.
    SPEC-10: Streamlined with clear error handling.
    """
    if not states:
        return {"error": "No states provided"}
    
    try:
        total_conversions = sum(state.converted_temps for state in states[1:])
        years_simulated = len(states) - 1
        theoretical_max = sum(state.annual_conversion_cap for state in states[1:] if state.annual_conversion_cap > 0)
        
        efficiency = total_conversions / theoretical_max if theoretical_max > 0 else 0.0
        
        return {
            'total_conversions': total_conversions,
            'theoretical_maximum': theoretical_max,
            'efficiency': efficiency,
            'years_simulated': years_simulated,
        }
        
    except Exception as e:
        logger.error(f"Error analyzing conversion queue efficiency: {e}")
        return {"error": str(e)}

def export_country_cap_analysis(states: List[SimulationState], output_path: str) -> None:
    """Export per-country cap analysis to CSV."""
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['year', 'total_workers', 'temporary_workers', 'conversions', 'cap_enabled'])
            
            for state in states:
                writer.writerow([
                    state.year,
                    state.total_workers,
                    state.temporary_workers,
                    state.converted_temps,
                    state.country_cap_enabled
                ])
        
        logger.info(f"Exported country cap analysis to {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting country cap analysis: {e}")
        raise

def ensure_output_directory(output_dir: str) -> Path:
    """Ensure output directory exists."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def safe_file_write(filepath: str, content: str, encoding: str = 'utf-8') -> bool:
    """Safely write content to file with error handling."""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing file {filepath}: {e}")
        return False

# SPEC-10: Removed redundant validation functions (validate impossible conditions
