# Also fix the CLI file - it has the same escaping issues
# src/simulation/cli.py
"""
Command-line interface for the workforce growth simulation.
Provides a user-friendly way to run simulations with various parameters.
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
    validate_configuration, format_number, format_percentage
)
from .empirical_params import (
    DEFAULT_YEARS, DEFAULT_SEED, H1B_SHARE, 
    ANNUAL_PERMANENT_ENTRY_RATE, ANNUAL_H1B_ENTRY_RATE,
    PYTHON_MIN_VERSION
)

def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Workforce Growth Simulation - Model permanent vs temporary worker dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.simulation.cli --initial-workers 10000
  python -m src.simulation.cli --initial-workers 50000 --years 20 --seed 123
  python -m src.simulation.cli --initial-workers 10000 --live-fetch --output results.csv
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
        help="Fetch latest workforce data from authoritative sources"
    )
    
    parser.add_argument(
        "--output",
        default="data/sample_output.csv",
        help="Output CSV file path (default: data/sample_output.csv)"
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

def print_simulation_header(config: SimulationConfig, live_data: Optional[dict] = None) -> None:
    """Print simulation header with parameters."""
    print("\\n" + "="*60)
    print("WORKFORCE GROWTH SIMULATION")
    print("="*60)
    print(f"Initial workforce: {format_number(config.initial_workers)}")
    print(f"Simulation years: {config.years}")
    print(f"Random seed: {config.seed or 'None (random)'}")
    print(f"Output file: {config.output_path}")
    
    if live_data:
        print("\\nUsing live-fetched data:")
        print(f"  H-1B share: {format_percentage(live_data.get('h1b_share', H1B_SHARE))}")
        print(f"  Annual H-1B entry rate: {format_percentage(live_data.get('annual_h1b_entry_rate', ANNUAL_H1B_ENTRY_RATE))}")
    else:
        print("\\nUsing default empirical parameters:")
        print(f"  H-1B share: {format_percentage(H1B_SHARE)}")
        print(f"  Annual permanent entry rate: {format_percentage(ANNUAL_PERMANENT_ENTRY_RATE)}")
        print(f"  Annual H-1B entry rate: {format_percentage(ANNUAL_H1B_ENTRY_RATE)}")
    
    print("="*60)

def print_simulation_results(simulation: Simulation) -> None:
    """Print summary of simulation results."""
    stats = simulation.get_summary_stats()
    
    print("\\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Years simulated: {stats['years_simulated']}")
    print(f"Initial workforce: {format_number(stats['initial_workforce'])}")
    print(f"Final workforce: {format_number(stats['final_workforce'])}")
    print(f"Total growth: {format_number(stats['total_growth'])} workers")
    print(f"Average annual growth rate: {format_percentage(stats['average_annual_growth_rate'])}")
    
    print("\\nWorkforce composition:")
    print(f"  Initial H-1B share: {format_percentage(stats['initial_h1b_share'])}")
    print(f"  Final H-1B share: {format_percentage(stats['final_h1b_share'])}")
    print(f"  H-1B share change: {format_percentage(stats['h1b_share_change'], 3)}")
    
    print("\\nNew workers added:")
    print(f"  Total permanent: {format_number(stats['total_new_permanent'])}")
    print(f"  Total temporary: {format_number(stats['total_new_temporary'])}")
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
        
        # Create configuration
        config = SimulationConfig(
            initial_workers=args.initial_workers,
            years=args.years,
            seed=args.seed,
            live_fetch=args.live_fetch,
            output_path=args.output
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
        if config.live_fetch:
            if not args.quiet:
                print("Fetching live workforce data...")
            live_data = fetch_live_data()
            if live_data:
                if not args.quiet:
                    print("Successfully fetched live data")
            else:
                print("Warning: Failed to fetch live data, using defaults")
        
        # Print header (unless quiet mode)
        if not args.quiet:
            print_simulation_header(config, live_data)
        
        # Run simulation
        simulation = Simulation(config)
        states = simulation.run()
        
        # Save results
        save_simulation_results(states, config.output_path)
        
        # Print results (unless quiet mode)
        if not args.quiet:
            print_simulation_results(simulation)
            print_data_sources(live_data)
        
        # Validation check
        if not simulation.validate_proportional_growth():
            logger.warning("Proportional growth validation failed - check parameters")
        
        if not args.quiet:
            print(f"\\nSimulation completed successfully!")
            print(f"Results saved to: {config.output_path}")
        
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

