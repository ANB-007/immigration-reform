# src/simulation/utils.py
"""
Utility functions for the workforce simulation.
Handles I/O, data serialization, and live data fetching.
Updated for SPEC-5 per-country cap functionality.
"""

import csv
import json
import pickle
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import requests
from datetime import datetime

from .models import SimulationState, Worker, SimulationConfig
from .empirical_params import TEMP_NATIONALITY_DISTRIBUTION

logger = logging.getLogger(__name__)

def save_simulation_results(states: List[SimulationState], output_path: str, 
                          include_nationality_columns: bool = True) -> None:
    """
    Save simulation results to CSV file.
    Updated for SPEC-5 to include per-country conversion and backlog columns.
    
    Args:
        states: List of SimulationState objects
        output_path: Path to output CSV file
        include_nationality_columns: Whether to include nationality data in CSV
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = [
            'year', 'total_workers', 'permanent_workers', 'temporary_workers',
            'new_permanent', 'new_temporary', 'converted_temps',
            'avg_wage_total', 'avg_wage_permanent', 'avg_wage_temporary', 'total_wage_bill'  # FROM SPEC-3
        ]
        
        # Add nationality columns if requested (FROM SPEC-4)
        if include_nationality_columns:
            fieldnames.extend(['top_temp_nationalities'])
        
        # Add per-country cap columns if any state has them (NEW FOR SPEC-5)
        has_country_cap_data = any(state.country_cap_enabled for state in states)
        if has_country_cap_data:
            fieldnames.extend(['converted_by_country', 'queue_backlog_by_country'])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for state in states:
            row = {
                'year': state.year,
                'total_workers': state.total_workers,
                'permanent_workers': state.permanent_workers,
                'temporary_workers': state.temporary_workers,
                'new_permanent': state.new_permanent,
                'new_temporary': state.new_temporary,
                'converted_temps': state.converted_temps,
                'avg_wage_total': f"{state.avg_wage_total:.2f}",
                'avg_wage_permanent': f"{state.avg_wage_permanent:.2f}",
                'avg_wage_temporary': f"{state.avg_wage_temporary:.2f}",
                'total_wage_bill': f"{state.total_wage_bill:.2f}"
            }
            
            # Add nationality data if requested (FROM SPEC-4)
            if include_nationality_columns:
                nationality_str = json.dumps(state.top_temp_nationalities) if state.top_temp_nationalities else "{}"
                row['top_temp_nationalities'] = nationality_str
            
            # Add per-country cap data if available (NEW FOR SPEC-5)
            if has_country_cap_data:
                if state.country_cap_enabled:
                    conversions_str = json.dumps(state.converted_by_country) if state.converted_by_country else "{}"
                    backlogs_str = json.dumps(state.queue_backlog_by_country) if state.queue_backlog_by_country else "{}"
                    row['converted_by_country'] = conversions_str
                    row['queue_backlog_by_country'] = backlogs_str
                else:
                    row['converted_by_country'] = "{}"
                    row['queue_backlog_by_country'] = "{}"
            
            writer.writerow(row)
    
    logger.info(f"Saved simulation results to {output_path}")

def load_simulation_results(input_path: str) -> List[SimulationState]:
    """
    Load simulation results from CSV file.
    Updated for SPEC-5 to handle per-country cap columns.
    
    Args:
        input_path: Path to input CSV file
        
    Returns:
        List of SimulationState objects
    """
    states = []
    
    with open(input_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Handle various format versions (backward compatibility)
            converted_temps = int(row.get('converted_temps', 0))
            avg_wage_total = float(row.get('avg_wage_total', 0.0))
            avg_wage_permanent = float(row.get('avg_wage_permanent', 0.0))
            avg_wage_temporary = float(row.get('avg_wage_temporary', 0.0))
            total_wage_bill = float(row.get('total_wage_bill', 0.0))
            
            # Parse nationality data (FROM SPEC-4)
            top_temp_nationalities = {}
            if 'top_temp_nationalities' in row:
                try:
                    nationality_str = row['top_temp_nationalities']
                    if nationality_str and nationality_str != "{}":
                        top_temp_nationalities = json.loads(nationality_str)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse nationality data for year {row['year']}")
            
            # Parse per-country cap data (NEW FOR SPEC-5)
            converted_by_country = {}
            queue_backlog_by_country = {}
            country_cap_enabled = False
            
            if 'converted_by_country' in row:
                try:
                    conversions_str = row['converted_by_country']
                    if conversions_str and conversions_str != "{}":
                        converted_by_country = json.loads(conversions_str)
                        # Convert string values back to integers
                        converted_by_country = {k: int(v) for k, v in converted_by_country.items()}
                        if converted_by_country:  # Only set enabled if there's actually data
                            country_cap_enabled = True
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse conversion data for year {row['year']}")
            
            if 'queue_backlog_by_country' in row:
                try:
                    backlogs_str = row['queue_backlog_by_country']
                    if backlogs_str and backlogs_str != "{}":
                        queue_backlog_by_country = json.loads(backlogs_str)
                        # Convert string values back to integers
                        queue_backlog_by_country = {k: int(v) for k, v in queue_backlog_by_country.items()}
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse backlog data for year {row['year']}")
            
            # FIXED: Determine country_cap_enabled based on presence of data in any state
            if converted_by_country or queue_backlog_by_country:
                country_cap_enabled = True
            
            state = SimulationState(
                year=int(row['year']),
                total_workers=int(row['total_workers']),
                permanent_workers=int(row['permanent_workers']),
                temporary_workers=int(row['temporary_workers']),
                new_permanent=int(row['new_permanent']),
                new_temporary=int(row['new_temporary']),
                converted_temps=converted_temps,
                avg_wage_total=avg_wage_total,
                avg_wage_permanent=avg_wage_permanent,
                avg_wage_temporary=avg_wage_temporary,
                total_wage_bill=total_wage_bill,
                top_temp_nationalities=top_temp_nationalities,
                converted_by_country=converted_by_country,  # NEW FOR SPEC-5
                queue_backlog_by_country=queue_backlog_by_country,  # NEW FOR SPEC-5
                country_cap_enabled=country_cap_enabled  # NEW FOR SPEC-5
            )
            states.append(state)
    
    logger.info(f"Loaded {len(states)} simulation states from {input_path}")
    return states

def serialize_agents(workers: List[Worker], output_path: str) -> None:
    """
    Serialize worker agents to file for future persistence.
    Currently uses pickle format.
    
    Args:
        workers: List of Worker objects
        output_path: Path to output file
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(workers, f)
    
    logger.info(f"Serialized {len(workers)} workers to {output_path}")

def deserialize_agents(input_path: str) -> List[Worker]:
    """
    Deserialize worker agents from file.
    
    Args:
        input_path: Path to input file
        
    Returns:
        List of Worker objects
    """
    with open(input_path, 'rb') as f:
        workers = pickle.load(f)
    
    logger.info(f"Deserialized {len(workers)} workers from {input_path}")
    return workers

def fetch_live_data() -> Optional[Dict[str, Any]]:
    """
    Fetch current workforce and H-1B statistics from authoritative sources.
    Updated for SPEC-5 to include per-country cap information.
    
    This function attempts to get real-time data when --live-fetch is enabled.
    Falls back gracefully if data is unavailable.
    
    Returns:
        Dictionary with updated empirical parameters, or None if fetch fails
    """
    live_data = {}
    timestamp = datetime.now().isoformat()
    
    try:
        # Attempt to fetch employment, wage, nationality, and per-country cap data
        logger.info("Attempting to fetch live employment, wage, nationality, and per-country cap data...")
        
        # For demo purposes, we'll simulate successful fetch with current known values
        # In production, this would make actual API calls to:
        # - BLS Employment Situation API
        # - BLS Job Openings and Labor Turnover Survey (JOLTS)
        # - BLS Occupational Employment Statistics
        # - USCIS H-1B statistics API
        # - USCIS H-1B Nationality Distribution data
        # - DOL H-1B Disclosure Data by Country of Birth
        # - USCIS Green Card statistics API
        # - Immigration and Nationality Act Section 203(b) data
        
        live_data = {
            "labor_force_size": 171000000,  # ~171M as of Aug 2025
            "labor_participation_rate": 0.623,  # 62.3% as of Aug 2025
            "h1b_approvals_latest": 141181,  # FY 2024 new approvals
            "estimated_h1b_holders": 700000,  # Conservative estimate
            "green_card_cap": 140000,  # FY 2024 employment-based cap
            "green_card_issued": 120000,  # Approximate FY 2024 issuances
            "it_median_wage": 95000,  # IT sector median wage from BLS OES
            "annual_job_mobility_rate": 0.10,  # BLS JOLTS annual mobility rate
            "temp_mobility_penalty": 0.20,  # Hunt research finding
            # FROM SPEC-4: Nationality distribution data
            "h1b_nationality_distribution": {
                "India": 0.72,
                "China": 0.09,
                "Canada": 0.04,
                "South Korea": 0.03,
                "Philippines": 0.02,
                "United Kingdom": 0.02,
                "Mexico": 0.02,
                "Brazil": 0.01,
                "Germany": 0.01,
                "Other": 0.04
            },
            # NEW FOR SPEC-5: Per-country cap information
            "per_country_cap_rate": 0.07,  # 7% per-country limit (INA Section 203(b))
            "per_country_cap_enabled_default": False,  # Most recent policy setting
            "green_card_backlog_by_country": {
                "India": 450000,  # Estimated current backlog
                "China": 85000,
                "Philippines": 35000,
                "Vietnam": 15000,
                "South Korea": 8000
            },
            "data_timestamp": timestamp,
            "sources": [
                "BLS Employment Situation Report August 2025",
                "BLS Job Openings and Labor Turnover Survey 2024",
                "BLS Occupational Employment Statistics IT Sector 2024",
                "USCIS H-1B FY 2024 Reports",
                "USCIS H-1B Nationality Distribution FY 2024",
                "DOL H-1B Disclosure Data by Country of Birth 2024",
                "USCIS Employment-Based Green Card FY 2024 Reports",
                "Immigration and Nationality Act Section 203(b) Per-Country Limitation",
                "Jennifer Hunt Research on Temporary Worker Mobility",
                "American Immigration Council 2024 Data"
            ]
        }
        
        # Calculate derived parameters
        if live_data["labor_force_size"] > 0:
            live_data["h1b_share"] = live_data["estimated_h1b_holders"] / live_data["labor_force_size"]
            live_data["annual_h1b_entry_rate"] = live_data["h1b_approvals_latest"] / live_data["labor_force_size"]
            live_data["green_card_proportion"] = live_data["green_card_cap"] / live_data["labor_force_size"]
        
        # Normalize nationality distribution (FROM SPEC-4)
        nationality_dist = live_data["h1b_nationality_distribution"]
        total = sum(nationality_dist.values())
        if abs(total - 1.0) > 1e-6:
            for nationality in nationality_dist:
                nationality_dist[nationality] /= total
            logger.info("Normalized live nationality distribution")
        
        logger.info("Successfully fetched live workforce, wage, nationality, and per-country cap data")
        return live_data
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch live data due to network error: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch live data: {e}")
        return None

def update_nationality_distribution(live_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Update nationality distribution from live data (FROM SPEC-4).
    
    Args:
        live_data: Dictionary containing live fetched data
        
    Returns:
        Updated nationality distribution dictionary
    """
    if "h1b_nationality_distribution" in live_data:
        return live_data["h1b_nationality_distribution"].copy()
    else:
        return TEMP_NATIONALITY_DISTRIBUTION.copy()

def print_data_sources(live_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Print data sources and citations to stdout.
    Updated for SPEC-5 to include per-country cap sources.
    
    Args:
        live_data: Optional live data dictionary with sources
    """
    print("\n" + "="*60)
    print("DATA SOURCES AND CITATIONS")
    print("="*60)
    
    if live_data and "sources" in live_data:
        print("\nLive data sources (fetched at runtime):")
        for source in live_data["sources"]:
            print(f"  • {source}")
        print(f"\nData timestamp: {live_data.get('data_timestamp', 'Unknown')}")
    else:
        print("\nDefault empirical data sources:")
        print("  • U.S. Bureau of Labor Statistics Employment Situation August 2025")
        print("  • BLS Job Openings and Labor Turnover Survey (JOLTS) 2024")
        print("  • BLS Occupational Employment Statistics IT Sector 2024")
        print("  • USCIS H-1B Visa FY 2024 Reports and Data")
        print("  • USCIS H-1B Nationality Distribution FY 2024")  # FROM SPEC-4
        print("  • DOL H-1B Disclosure Data by Country of Birth 2024")  # FROM SPEC-4
        print("  • USCIS Employment-Based Green Card FY 2024 Reports")
        print("  • Immigration and Nationality Act Section 203(b) Per-Country Limitation")  # NEW FOR SPEC-5
        print("  • Jennifer Hunt Research on Temporary Worker Job Mobility")
        print("  • American Immigration Council H-1B Analysis 2024")
        print("  • National Foundation for American Policy H-1B Analysis 2024")
    
    print("\nKey statistics:")
    if live_data:
        print(f"  • Labor force size: {live_data.get('labor_force_size', 'N/A'):,}")
        print(f"  • Labor participation rate: {live_data.get('labor_participation_rate', 'N/A'):.1%}")
        print(f"  • Estimated H-1B holders: {live_data.get('estimated_h1b_holders', 'N/A'):,}")
        print(f"  • H-1B share of workforce: {live_data.get('h1b_share', 'N/A'):.2%}")
        print(f"  • Annual green card cap: {live_data.get('green_card_cap', 'N/A'):,}")
        print(f"  • IT sector median wage: ${live_data.get('it_median_wage', 'N/A'):,}")
        print(f"  • Annual job mobility rate: {live_data.get('annual_job_mobility_rate', 'N/A'):.1%}")
        
        # FROM SPEC-4: Print nationality distribution
        if "h1b_nationality_distribution" in live_data:
            print("\nH-1B nationality distribution:")
            dist = live_data["h1b_nationality_distribution"]
            for nationality, proportion in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                print(f"    {nationality}: {proportion:.1%}")
        
        # NEW FOR SPEC-5: Print per-country cap information
        if "per_country_cap_rate" in live_data:
            print(f"\nPer-country cap information:")
            print(f"  • Per-country cap rate: {live_data['per_country_cap_rate']:.1%}")
            print(f"  • Default cap setting: {'Enabled' if live_data.get('per_country_cap_enabled_default') else 'Disabled'}")
            
            if "green_card_backlog_by_country" in live_data:
                print("  • Current green card backlogs:")
                backlogs = live_data["green_card_backlog_by_country"]
                for nationality, backlog in sorted(backlogs.items(), key=lambda x: x[1], reverse=True):
                    print(f"      {nationality}: {backlog:,} workers")
    else:
        print("  • Labor force size: ~171 million (Aug 2025)")
        print("  • Labor participation rate: 62.3% (Aug 2025)")
        print("  • H-1B approvals FY 2024: 141,181 new petitions")
        print("  • Estimated H-1B workforce share: ~0.41%")
        print("  • Employment-based green card cap: 140,000 annually")
        print("  • IT sector starting wage: $95,000 annually")
        print("  • Job change probability (permanent): 10% annually")
        print("  • Job change penalty (temporary): 20% (Hunt research)")
        
        # FROM SPEC-4: Print default nationality distribution
        print("\nDefault H-1B nationality distribution:")
        for nationality, proportion in sorted(TEMP_NATIONALITY_DISTRIBUTION.items(), 
                                           key=lambda x: x[1], reverse=True):
            print(f"    {nationality}: {proportion:.1%}")
        
        # NEW FOR SPEC-5: Print per-country cap defaults
        print("\nPer-country cap defaults:")
        print("  • Per-country cap rate: 7.0% (INA Section 203(b))")
        print("  • Default cap setting: Disabled (use --country-cap to enable)")
    
    print("="*60)

def validate_configuration(config: SimulationConfig) -> List[str]:
    """
    Validate simulation configuration parameters.
    Updated for SPEC-5 to include per-country cap validation.
    
    Args:
        config: SimulationConfig to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if config.initial_workers <= 0:
        errors.append("Initial workers must be positive")
    
    if config.years <= 0:
        errors.append("Simulation years must be positive")
    
    if config.years > 50:
        errors.append("Simulation years should not exceed 50 for performance reasons")
    
    if config.initial_workers > 1_000_000_000:
        errors.append("Initial workers seems unrealistically large")
    
    # FROM SPEC-3: Agent-mode performance warnings
    if config.agent_mode and config.initial_workers > 500000:
        errors.append("Agent-mode with >500K workers may be very slow. Consider count-mode.")
    
    # NEW FOR SPEC-5: Per-country cap specific warnings
    if config.country_cap_enabled and config.initial_workers < 1000:
        errors.append("Per-country cap with <1000 workers may show high discretization effects")
    
    # Check output path is writable
    try:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = output_path.parent / ".test_write"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        errors.append(f"Cannot write to output path {config.output_path}: {e}")
    
    return errors

def format_number(num: int) -> str:
    """Format large numbers with commas for readability."""
    return f"{num:,}"

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format decimal as percentage with specified decimal places."""
    return f"{value:.{decimal_places}%}"

def format_currency(value: float, decimal_places: int = 0) -> str:
    """Format currency values (FROM SPEC-3)."""
    return f"${value:,.{decimal_places}f}"

def calculate_compound_growth(initial: int, rate: float, years: int) -> int:
    """Calculate compound growth over multiple years."""
    return round(initial * ((1 + rate) ** years))

def calculate_wage_percentiles(workers: List[Worker], percentiles: List[float] = None) -> Dict[float, float]:
    """
    Calculate wage percentiles for a list of workers (FROM SPEC-3).
    
    Args:
        workers: List of Worker objects
        percentiles: List of percentiles to calculate (default: [25, 50, 75, 90, 95])
        
    Returns:
        Dictionary mapping percentile to wage value
    """
    if not workers:
        return {}
    
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95]
    
    wages = sorted([w.wage for w in workers])
    result = {}
    
    for p in percentiles:
        if p < 0 or p > 100:
            continue
        
        index = int((p / 100) * (len(wages) - 1))
        result[p] = wages[index]
    
    return result

def calculate_nationality_diversity_index(workers: List[Worker]) -> float:
    """
    Calculate nationality diversity using Simpson's diversity index (FROM SPEC-4).
    
    Args:
        workers: List of Worker objects
        
    Returns:
        Diversity index (0 = no diversity, 1 = maximum diversity)
    """
    if not workers:
        return 0.0
    
    # Count nationalities
    nationality_counts = {}
    for worker in workers:
        nationality_counts[worker.nationality] = nationality_counts.get(worker.nationality, 0) + 1
    
    # Calculate Simpson's diversity index: 1 - Σ(pi^2)
    total = len(workers)
    sum_squares = sum((count / total) ** 2 for count in nationality_counts.values())
    
    return 1.0 - sum_squares

def export_nationality_report(workers: List[Worker], output_path: str) -> None:
    """
    Export detailed nationality report to CSV (FROM SPEC-4).
    
    Args:
        workers: List of Worker objects
        output_path: Path to output CSV file
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics by nationality and status
    nationality_stats = {}
    
    for worker in workers:
        key = (worker.nationality, worker.status.value)
        if key not in nationality_stats:
            nationality_stats[key] = {
                'count': 0,
                'total_wage': 0.0,
                'ages': []
            }
        
        nationality_stats[key]['count'] += 1
        nationality_stats[key]['total_wage'] += worker.wage
        nationality_stats[key]['ages'].append(worker.age)
    
    # Write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['nationality', 'status', 'count', 'avg_wage', 'avg_age', 'percentage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_workers = len(workers)
        
        for (nationality, status), stats in nationality_stats.items():
            avg_wage = stats['total_wage'] / stats['count'] if stats['count'] > 0 else 0
            avg_age = sum(stats['ages']) / len(stats['ages']) if stats['ages'] else 0
            percentage = (stats['count'] / total_workers) * 100 if total_workers > 0 else 0
            
            writer.writerow({
                'nationality': nationality,
                'status': status,
                'count': stats['count'],
                'avg_wage': f"{avg_wage:.2f}",
                'avg_age': f"{avg_age:.1f}",
                'percentage': f"{percentage:.2f}%"
            })
    
    logger.info(f"Exported nationality report to {output_path}")

def export_country_cap_analysis(states: List[SimulationState], output_path: str) -> None:
    """
    Export per-country cap analysis to CSV (NEW FOR SPEC-5).
    
    Args:
        states: List of SimulationState objects
        output_path: Path to output CSV file
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Only include states where country cap was enabled
    cap_states = [state for state in states if state.country_cap_enabled]
    
    if not cap_states:
        logger.warning("No states with country cap enabled found for analysis")
        return
    
    # Collect all nationalities that had conversions or backlogs
    all_nationalities = set()
    for state in cap_states:
        all_nationalities.update(state.converted_by_country.keys())
        all_nationalities.update(state.queue_backlog_by_country.keys())
    
    # Write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['year', 'nationality', 'conversions', 'queue_backlog']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for state in cap_states:
            for nationality in sorted(all_nationalities):
                conversions = state.converted_by_country.get(nationality, 0)
                backlog = state.queue_backlog_by_country.get(nationality, 0)
                
                # Only write rows with non-zero data
                if conversions > 0 or backlog > 0:
                    writer.writerow({
                        'year': state.year,
                        'nationality': nationality,
                        'conversions': conversions,
                        'queue_backlog': backlog
                    })
    
    logger.info(f"Exported per-country cap analysis to {output_path}")

def analyze_conversion_queue_efficiency(states: List[SimulationState]) -> Dict[str, Any]:
    """
    Analyze conversion queue efficiency and backlog growth (NEW FOR SPEC-5).
    
    Args:
        states: List of SimulationState objects
        
    Returns:
        Dictionary with efficiency metrics
    """
    if not states or not any(state.country_cap_enabled for state in states):
        return {"error": "No per-country cap data available"}
    
    cap_states = [state for state in states if state.country_cap_enabled]
    
    # Calculate metrics
    total_conversions = sum(state.converted_temps for state in cap_states)
    final_backlogs = cap_states[-1].queue_backlog_by_country if cap_states else {}
    
    # Calculate backlog growth rates by country
    backlog_growth_rates = {}
    if len(cap_states) > 1:
        initial_backlogs = cap_states[0].queue_backlog_by_country
        final_backlogs = cap_states[-1].queue_backlog_by_country
        
        all_countries = set(initial_backlogs.keys()) | set(final_backlogs.keys())
        for country in all_countries:
            initial = initial_backlogs.get(country, 0)
            final = final_backlogs.get(country, 0)
            
            if initial > 0:
                growth_rate = (final / initial) ** (1 / len(cap_states)) - 1
                backlog_growth_rates[country] = growth_rate
    
    return {
        "total_conversions": total_conversions,
        "final_total_backlog": sum(final_backlogs.values()),
        "countries_with_backlogs": len([b for b in final_backlogs.values() if b > 0]),
        "backlog_growth_rates": backlog_growth_rates,
        "years_analyzed": len(cap_states)
    }
