# src/simulation/cli.py
"""
Command-line interface for the workforce growth simulation.
Provides a user-friendly way to run simulations with various parameters.
Updated for SPEC-5 per-country cap functionality.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from .models import SimulationConfig
from .sim import Simulation
from .utils import (
    save_simulation_results, fetch_live_data, print_data_sources,
    validate_configuration, format_number, format_percentage, format_currency,
    update_nationality_distribution, export_nationality_report
)
from .empirical_params import (
    DEFAULT_YEARS, DEFAULT_SEED, H1B_SHARE, 
    ANNUAL_PERMANENT_ENTRY_RATE, ANNUAL_H1B_ENTRY_RATE,
    PYTHON_MIN_VERSION, GREEN_CARD_CAP_ABS, REAL_US_WORKFORCE_SIZE,
    STARTING_WAGE, JOB_CHANGE_PROB_PERM, TEMP_JOB_CHANGE_PENALTY,
    WAGE_JUMP_FACTOR_MEAN, INDUSTRY_NAME, TEMP_NATIONALITY_DISTRIBUTION,
    PERMANENT_NATIONALITY, PER_COUNTRY_CAP_SHARE, ENABLE_COUNTRY_CAP
)

def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Workforce Growth Simulation - Model permanent vs temporary worker dynamics with wage tracking, green card conversions, nationality segmentation, and per-country caps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.simulation.cli --initial-workers 10000
  python -m src.simulation.cli --initial-workers 50000 --years 20 --seed 123
  python -m src.simulation.cli --initial-workers 10000 --live-fetch --output results.csv
  python -m src.simulation.cli --initial-workers 100000 --count-mode  # For large simulations
  python -m src.simulation.cli --initial-workers 10000 --show-nationality-summary  # Show nationality breakdown
  python -m src.simulation.cli --initial-workers 10000 --country-cap  # Enable 7%% per-country cap
  python -m src.simulation.cli --initial-workers 10000 --no-country-cap  # Disable cap (default)
        """
    )
    
    parser.add_argument(
        "--initial-workers",
        type=int,
        required=True,
        help="Starting number of worker agents (required)"
    )
    
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help=f"Number of years to simulate (default: {DEFAULT_YEARS})"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (optional)"
    )
    
    parser.add_argument(
        "--live-fetch",
        action="store_true",
        help="Fetch latest workforce and nationality data from authoritative sources"
    )
    
    parser.add_argument(
        "--output",
        default="data/sample_output.csv",
        help="Output CSV file path (default: data/sample_output.csv)"
    )
    
    # FROM SPEC-3: Agent-mode vs count-mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--agent-mode",
        action="store_true",
        default=True,
        help="Use agent-based simulation with individual wage and nationality tracking (default)"
    )
    mode_group.add_argument(
        "--count-mode", 
        action="store_true",
        help="Use count-based simulation for large populations (faster, approximate wages/nationalities)"
    )
    
    # FROM SPEC-4: Nationality summary option
    parser.add_argument(
        "--show-nationality-summary",
        action="store_true",
        help="Show nationality breakdown at start and end of simulation"
    )
    
    parser.add_argument(
        "--export-nationality-report",
        help="Export detailed nationality report to specified CSV file (agent-mode only)"
    )
    
    # NEW FOR SPEC-5: Per-country cap options
    cap_group = parser.add_mutually_exclusive_group()
    cap_group.add_argument(
        "--country-cap",
        action="store_true",
        help="Enable 7%% per-country limit on employment-based green cards"  # FIXED: Escaped %
    )
    cap_group.add_argument(
        "--no-country-cap",
        action="store_true", 
        default=True,
        help="Disable per-country cap (default - use global FIFO queue)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    return parser

def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity settings."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def print_simulation_header(config: SimulationConfig, live_data: Optional[dict] = None,
                          nationality_distribution: Optional[dict] = None) -> None:
    """
    Print simulation header with parameters.
    Updated for SPEC-5 to show per-country cap information.
    """
    print("\n" + "="*60)
    print("WORKFORCE GROWTH SIMULATION")
    print("="*60)
    print(f"Industry: {INDUSTRY_NAME}")
    print(f"Initial workforce: {format_number(config.initial_workers)}")
    print(f"Simulation years: {config.years}")
    print(f"Simulation mode: {'Agent-based' if config.agent_mode else 'Count-based'}")
    print(f"Random seed: {config.seed or 'None (random)'}")
    print(f"Output file: {config.output_path}")
    
    # Performance warning for large agent-mode simulations
    if config.agent_mode and config.initial_workers > 200000:
        print(f"\n⚠️  WARNING: Agent-mode with {format_number(config.initial_workers)} workers will be slow.")
        print("   Consider using --count-mode for large simulations.")
    
    # Green card conversion cap (FROM SPEC-2)
    cap_proportion = GREEN_CARD_CAP_ABS / REAL_US_WORKFORCE_SIZE
    annual_cap = round(config.initial_workers * cap_proportion)
    print(f"\nGreen card conversion system:")
    print(f"  Annual conversions: {annual_cap} ({format_percentage(cap_proportion, 4)})")
    print(f"  Based on US cap: {format_number(GREEN_CARD_CAP_ABS)} / {format_number(REAL_US_WORKFORCE_SIZE)}")
    
    # NEW FOR SPEC-5: Per-country cap information
    if config.country_cap_enabled:
        per_country_cap = round(annual_cap * PER_COUNTRY_CAP_SHARE)
        # FIXED: Use format_percentage instead of raw % signs
        print(f"  Per-country cap: ENABLED ({format_percentage(PER_COUNTRY_CAP_SHARE)})")
        print(f"  Max per country: {per_country_cap} conversions/year")
        print(f"  Queue mode: Separate nationality queues with FIFO within each")
    else:
        print(f"  Per-country cap: DISABLED")
        print(f"  Queue mode: Single global FIFO queue")
    
    # FROM SPEC-3: Wage and job mobility parameters
    temp_job_change_prob = JOB_CHANGE_PROB_PERM * (1 - TEMP_JOB_CHANGE_PENALTY)
    print(f"\nWage and job mobility parameters:")
    print(f"  Starting wage: {format_currency(STARTING_WAGE)}")
    print(f"  Job change probability (permanent): {format_percentage(JOB_CHANGE_PROB_PERM)}")
    print(f"  Job change probability (temporary): {format_percentage(temp_job_change_prob)}")
    print(f"  Average wage jump on job change: {format_percentage(WAGE_JUMP_FACTOR_MEAN - 1)}")
    
    # FROM SPEC-4: Nationality distribution
    print(f"\nNationality distribution:")
    print(f"  Permanent workers: {PERMANENT_NATIONALITY}")
    if nationality_distribution:
        print("  Temporary worker nationalities (top 5):")
        sorted_nationalities = sorted(nationality_distribution.items(), key=lambda x: x[1], reverse=True)
        for nationality, proportion in sorted_nationalities[:5]:
            print(f"    {nationality}: {format_percentage(proportion)}")
    else:
        print("  Temporary worker nationalities (top 5):")
        sorted_nationalities = sorted(TEMP_NATIONALITY_DISTRIBUTION.items(), key=lambda x: x[1], reverse=True)
        for nationality, proportion in sorted_nationalities[:5]:
            print(f"    {nationality}: {format_percentage(proportion)}")
    
    if live_data:
        print("\nUsing live-fetched data:")
        print(f"  H-1B share: {format_percentage(live_data.get('h1b_share', H1B_SHARE))}")
        print(f"  Annual H-1B entry rate: {format_percentage(live_data.get('annual_h1b_entry_rate', ANNUAL_H1B_ENTRY_RATE))}")
        if 'green_card_cap' in live_data:
            print(f"  Green card cap: {format_number(live_data['green_card_cap'])}")
        if 'it_median_wage' in live_data:
            print(f"  IT sector median wage: {format_currency(live_data['it_median_wage'])}")
        if 'annual_job_mobility_rate' in live_data:
            print(f"  Job mobility rate: {format_percentage(live_data['annual_job_mobility_rate'])}")
    else:
        print("\nUsing default empirical parameters:")
        print(f"  H-1B share: {format_percentage(H1B_SHARE)}")
        print(f"  Annual permanent entry rate: {format_percentage(ANNUAL_PERMANENT_ENTRY_RATE)}")
        print(f"  Annual H-1B entry rate: {format_percentage(ANNUAL_H1B_ENTRY_RATE)}")
    
    print("="*60)

def print_simulation_results(simulation: Simulation) -> None:
    """
    Print summary of simulation results.
    Updated for SPEC-5 to show per-country cap results.
    """
    stats = simulation.get_summary_stats()
    
    print("\\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Years simulated: {stats['years_simulated']}")
    print(f"Simulation mode: {stats['simulation_mode']}")
    print(f"Industry: {stats['industry']}")
    print(f"Initial workforce: {format_number(stats['initial_workforce'])}")
    print(f"Final workforce: {format_number(stats['final_workforce'])}")
    print(f"Total growth: {format_number(stats['total_growth'])} workers")
    print(f"Average annual growth rate: {format_percentage(stats['average_annual_growth_rate'])}")
    
    print("\\nWorkforce composition:")
    print(f"  Initial H-1B share: {format_percentage(stats['initial_h1b_share'])}")
    print(f"  Final H-1B share: {format_percentage(stats['final_h1b_share'])}")
    print(f"  H-1B share change: {format_percentage(stats['h1b_share_change'], 3)}")
    
    print("\\nWorker flows:")
    print(f"  Total permanent entries: {format_number(stats['total_new_permanent'])}")
    print(f"  Total temporary entries: {format_number(stats['total_new_temporary'])}")
    print(f"  Total conversions (temp→perm): {format_number(stats['total_conversions'])}")
    
    print("\\nConversion statistics:")
    print(f"  Annual conversion cap: {format_number(stats['annual_conversion_cap'])}")
    print(f"  Cap utilization: {format_percentage(stats['conversion_utilization'])}")
    
    # NEW FOR SPEC-5: Per-country cap results
    if stats.get('country_cap_enabled'):
        print(f"\\nPer-country cap results:")
        print(f"  Per-country cap: {format_number(stats['per_country_cap'])} ({format_percentage(stats['per_country_cap_rate'])})")
        
        if 'total_conversions_by_country' in stats:
            print("  Total conversions by country:")
            conversions_by_country = stats['total_conversions_by_country']
            sorted_conversions = sorted(conversions_by_country.items(), key=lambda x: x[1], reverse=True)
            for nationality, conversions in sorted_conversions[:5]:  # Top 5
                print(f"    {nationality}: {format_number(conversions)}")
        
        if 'final_queue_backlogs' in stats:
            print("  Final queue backlogs:")
            backlogs = stats['final_queue_backlogs']
            total_backlog = sum(backlogs.values())
            print(f"    Total backlog: {format_number(total_backlog)} workers")
            
            if backlogs:
                sorted_backlogs = sorted(backlogs.items(), key=lambda x: x[1], reverse=True)
                for nationality, backlog in sorted_backlogs[:5]:  # Top 5
                    if backlog > 0:
                        print(f"    {nationality}: {format_number(backlog)} workers")
        
        countries_with_backlogs = stats.get('countries_with_backlogs', 0)
        print(f"  Countries with backlogs: {countries_with_backlogs}")
    else:
        print("\\nPer-country cap: DISABLED (global FIFO queue used)")
    
    # FROM SPEC-3: Wage statistics
    print("\\nWage statistics:")
    print(f"  Initial average wage: {format_currency(stats['initial_avg_wage'])}")
    print(f"  Final average wage: {format_currency(stats['final_avg_wage'])}")
    print(f"  Total wage growth: {format_currency(stats['total_wage_growth'])}")
    print(f"  Average annual wage growth rate: {format_percentage(stats['average_annual_wage_growth_rate'])}")
    print(f"  Final total wage bill: {format_currency(stats['final_wage_bill'])}")
    
    print("\\nJob mobility rates used:")
    print(f"  Permanent workers: {format_percentage(stats['job_change_prob_permanent'])}")
    print(f"  Temporary workers: {format_percentage(stats['job_change_prob_temporary'])}")
    
    # FROM SPEC-4: Nationality statistics
    if 'initial_temp_nationalities' in stats and 'final_temp_nationalities' in stats:
        print("\\nTemporary worker nationality changes:")
        print("  Initial top nationalities:")
        for nationality, proportion in stats['initial_temp_nationalities'].items():
            print(f"    {nationality}: {format_percentage(proportion)}")
        
        print("  Final top nationalities:")
        for nationality, proportion in stats['final_temp_nationalities'].items():
            print(f"    {nationality}: {format_percentage(proportion)}")
    
    print("="*60)

def main() -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger(__name__)
    
    try:
        # Check Python version
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        required_version = PYTHON_MIN_VERSION
        if python_version < required_version:
            print(f"Error: Python {required_version}+ required, found {python_version}")
            return 1
        
        # Determine country cap setting (NEW FOR SPEC-5)
        # CLI flag overrides default parameter
        if args.country_cap:
            country_cap_enabled = True
        elif args.no_country_cap:
            country_cap_enabled = False
        else:
            country_cap_enabled = ENABLE_COUNTRY_CAP
        
        # Create configuration
        config = SimulationConfig(
            initial_workers=args.initial_workers,
            years=args.years,
            seed=args.seed,
            live_fetch=args.live_fetch,
            output_path=args.output,
            agent_mode=not args.count_mode,  # Default to agent-mode unless count-mode specified
            show_nationality_summary=args.show_nationality_summary,
            country_cap_enabled=country_cap_enabled  # NEW FOR SPEC-5
        )
        
        # Validate configuration
        errors = validate_configuration(config)
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  • {error}")
            return 1
        
        # Fetch live data if requested
        live_data = None
        updated_nationality_distribution = None
        if config.live_fetch:
            if not args.quiet:
                print("Fetching live workforce, wage, and nationality data...")
            live_data = fetch_live_data()
            if live_data:
                if not args.quiet:
                    print("Successfully fetched live data")
                    # Print updated workforce size if different
                    if live_data.get('labor_force_size') != REAL_US_WORKFORCE_SIZE:
                        print(f"Updated workforce size: {format_number(live_data['labor_force_size'])}")
                
                # FROM SPEC-4: Update nationality distribution from live data
                updated_nationality_distribution = update_nationality_distribution(live_data)
                if updated_nationality_distribution != TEMP_NATIONALITY_DISTRIBUTION:
                    print("Updated nationality distribution from live data")
            else:
                print("Warning: Failed to fetch live data, using defaults")
        
        # Print header (unless quiet mode)
        if not args.quiet:
            print_simulation_header(config, live_data, updated_nationality_distribution)
        
        # Run simulation
        simulation = Simulation(config)
        
        # Update nationality distribution in simulation if live data was fetched
        if updated_nationality_distribution:
            simulation.temp_nationality_distribution = updated_nationality_distribution
            simulation._validate_nationality_distribution()
        
        states = simulation.run()
        
        # Save results (NEW FOR SPEC-5: always include nationality columns, now includes per-country data)
        save_simulation_results(states, config.output_path, include_nationality_columns=True)
        
        # Export nationality report if requested (FROM SPEC-4)
        if args.export_nationality_report:
            if config.agent_mode:
                workers = simulation.to_agent_model()
                export_nationality_report(workers, args.export_nationality_report)
                if not args.quiet:
                    print(f"Nationality report exported to: {args.export_nationality_report}")
            else:
                print("Warning: Nationality report export requires agent-mode")
        
        # Print results (unless quiet mode)
        if not args.quiet:
            print_simulation_results(simulation)
            print_data_sources(live_data)
        
        # Validation checks
        if not simulation.validate_proportional_growth():
            logger.warning("Proportional growth validation failed - check parameters")
        
        if not simulation.validate_conversion_consistency():
            logger.warning("Conversion consistency validation failed - check implementation")
        
        # FROM SPEC-3: Validate wage consistency
        if not simulation.validate_wage_consistency():
            logger.warning("Wage consistency validation failed - check wage calculations")
        
        # FROM SPEC-4: Validate nationality consistency
        if not simulation.validate_nationality_consistency():
            logger.warning("Nationality consistency validation failed - check nationality logic")
        
        # NEW FOR SPEC-5: Validate country cap consistency
        if not simulation.validate_country_cap_consistency():
            logger.warning("Country cap consistency validation failed - check per-country cap logic")
        
        if not args.quiet:
            print(f"\\nSimulation completed successfully!")
            print(f"Results saved to: {config.output_path}")
            print(f"\\nNext steps:")
            print(f"  • Analyze wage growth patterns by nationality in the output CSV")
            print(f"  • Compare permanent vs temporary worker wage trajectories by nationality") 
            print(f"  • Examine the impact of green card conversions on nationality composition")
            if config.country_cap_enabled:
                print(f"  • Analyze per-country conversion patterns and queue backlogs")
                print(f"  • Compare results with --no-country-cap to see impact of 7% rule")
            else:
                print(f"  • Try running with --country-cap to see impact of 7% per-country limit")
            print(f"  • Use --export-nationality-report for detailed nationality analysis")
        
        return 0
        
    except KeyboardInterrupt:
        print("\\nSimulation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
