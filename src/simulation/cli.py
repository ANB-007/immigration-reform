# src/simulation/cli.py
"""
Command-line interface for the workforce growth simulation.
Provides a user-friendly way to run simulations with various parameters.
Updated for SPEC-8 with debug support and corrected conversion logic.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from .models import SimulationConfig, BacklogAnalysis
from .sim import Simulation
from .utils import (
    save_simulation_results, fetch_live_data, print_data_sources,
    validate_configuration, format_number, format_percentage, format_currency,
    update_nationality_distribution, export_nationality_report,
    save_backlog_analysis, load_backlog_analysis
)
from .empirical_params import (
    DEFAULT_YEARS, DEFAULT_SEED, H1B_SHARE, 
    ANNUAL_PERMANENT_ENTRY_RATE, ANNUAL_H1B_ENTRY_RATE,
    PYTHON_MIN_VERSION, GREEN_CARD_CAP_ABS, REAL_US_WORKFORCE_SIZE,
    STARTING_WAGE, JOB_CHANGE_PROB_PERM, TEMP_JOB_CHANGE_PENALTY,
    WAGE_JUMP_FACTOR_MEAN_PERM, WAGE_JUMP_FACTOR_MEAN_TEMP, INDUSTRY_NAME, 
    TEMP_NATIONALITY_DISTRIBUTION, PERMANENT_NATIONALITY, PER_COUNTRY_CAP_SHARE, 
    ENABLE_COUNTRY_CAP, OUTPUT_DIR, ENABLE_VISUALIZATION
)

# FROM SPEC-6: Import visualization module
try:
    from .visualization import SimulationVisualizer, validate_dataframes, validate_backlog_dataframes
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    print(f"Warning: Visualization module not available: {e}")

def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Workforce Growth Simulation - Model permanent vs temporary worker dynamics with wage tracking, green card conversions, nationality segmentation, per-country caps, visualization, and backlog analysis (CORRECTED SPEC-8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.simulation.cli --initial-workers 10000
  python -m src.simulation.cli --initial-workers 50000 --years 20 --seed 123
  python -m src.simulation.cli --initial-workers 10000 --live-fetch --output results.csv
  python -m src.simulation.cli --initial-workers 100000 --count-mode  # For large simulations
  python -m src.simulation.cli --initial-workers 10000 --show-nationality-summary  # Show nationality breakdown
  python -m src.simulation.cli --initial-workers 10000 --country-cap  # Enable 7%% per-country cap
  python -m src.simulation.cli --initial-workers 10000 --visualize-results  # Generate comparison charts
  python -m src.simulation.cli --initial-workers 10000 --compare-backlogs  # Compare backlogs by nationality
  python -m src.simulation.cli --initial-workers 10000 --debug  # Enable debug output (SPEC-8)
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
    
    # FROM SPEC-5: Per-country cap options
    cap_group = parser.add_mutually_exclusive_group()
    cap_group.add_argument(
        "--country-cap",
        action="store_true",
        help="Enable 7%% per-country limit on employment-based green cards"
    )
    cap_group.add_argument(
        "--no-country-cap",
        action="store_true", 
        default=True,
        help="Disable per-country cap (default - use global FIFO queue)"
    )
    
    # FROM SPEC-6: Visualization options
    parser.add_argument(
        "--visualize-results",
        action="store_true",
        help="Generate wage and workforce comparison charts (runs both capped and uncapped scenarios)"
    )
    
    # FROM SPEC-7: Backlog analysis options
    parser.add_argument(
        "--compare-backlogs",
        action="store_true",
        help="Compare final-year backlogs by nationality between capped and uncapped simulations"
    )
    
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Directory for visualization outputs (default: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Generate data but skip displaying plots (useful for batch processing)"
    )
    
    # CORRECTED FOR SPEC-8: Debug option
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output showing conversion caps, residuals, and queue lengths"
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

def setup_logging(verbose: bool = False, quiet: bool = False, debug: bool = False) -> None:
    """Configure logging based on verbosity settings."""
    if quiet:
        level = logging.ERROR
    elif debug or verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def run_simulation_scenario(config: SimulationConfig, scenario_name: str, 
                          live_data: Optional[dict] = None,
                          updated_nationality_distribution: Optional[dict] = None,
                          quiet: bool = False) -> tuple[pd.DataFrame, BacklogAnalysis]:
    """
    Run a single simulation scenario and return results as DataFrame and BacklogAnalysis.
    Updated for SPEC-8 to include debug support.
    """
    if not quiet:
        cap_status = "WITH" if config.country_cap_enabled else "WITHOUT"
        debug_status = " (DEBUG MODE)" if config.debug else ""
        print(f"\n🔬 Running simulation {scenario_name} ({cap_status} per-country cap){debug_status}...")
    
    # Run simulation
    simulation = Simulation(config)
    
    # Update nationality distribution if live data was fetched
    if updated_nationality_distribution:
        simulation.temp_nationality_distribution = updated_nationality_distribution
        simulation._validate_nationality_distribution()
    
    states = simulation.run()
    
    # Save results to temporary CSV for visualization
    temp_output_path = f"temp_{scenario_name.lower().replace(' ', '_')}_results.csv"
    save_simulation_results(states, temp_output_path, include_nationality_columns=True)
    
    # Load as DataFrame
    results_df = pd.read_csv(temp_output_path)
    
    # Generate backlog analysis
    scenario_type = "capped" if config.country_cap_enabled else "uncapped"
    backlog_analysis = BacklogAnalysis.from_simulation(simulation, scenario_type)
    
    # Clean up temporary file
    import os
    try:
        os.remove(temp_output_path)
    except OSError:
        pass
    
    if not quiet:
        final_state = states[-1]
        print(f"✅ Completed {scenario_name}: {final_state.total_workers:,} workers, "
              f"${final_state.avg_wage_total:,.0f} avg wage, "
              f"{backlog_analysis.total_backlog:,} total backlog")
        
        # CORRECTED FOR SPEC-8: Show debug info if enabled
        if config.debug:
            print(f"   Debug: Annual conversion cap: {simulation.annual_sim_cap}")
            print(f"   Debug: Cumulative conversions: {simulation.cumulative_conversions:,}")
            print(f"   Debug: Final queue lengths - Uncapped: {simulation.get_total_backlog_uncapped()}, "
                  f"Capped: {simulation.get_total_backlog_capped()}")
    
    return results_df, backlog_analysis

def print_simulation_header(config: SimulationConfig, live_data: Optional[dict] = None,
                          nationality_distribution: Optional[dict] = None) -> None:
    """Print simulation header with parameters. Updated for SPEC-8."""
    print("\n" + "="*60)
    print("WORKFORCE GROWTH SIMULATION (CORRECTED SPEC-8)")
    print("="*60)
    print(f"Industry: {INDUSTRY_NAME}")
    print(f"Initial workforce: {format_number(config.initial_workers)}")
    print(f"Simulation years: {config.years}")
    print(f"Simulation mode: {'Agent-based' if config.agent_mode else 'Count-based'}")
    print(f"Random seed: {config.seed or 'None (random)'}")
    print(f"Output file: {config.output_path}")
    
    # CORRECTED FOR SPEC-8: Show debug status
    if config.debug:
        print(f"Debug mode: ENABLED")
    
    # Performance warning for large agent-mode simulations
    if config.agent_mode and config.initial_workers > 200000:
        print(f"\n⚠️  WARNING: Agent-mode with {format_number(config.initial_workers)} workers will be slow.")
        print("   Consider using --count-mode for large simulations.")
    
    # CORRECTED FOR SPEC-8: Show fixed conversion cap information
    from .empirical_params import calculate_annual_sim_cap
    annual_cap, residual = calculate_annual_sim_cap(config.initial_workers)
    cap_proportion = GREEN_CARD_CAP_ABS / REAL_US_WORKFORCE_SIZE
    
    print(f"\nFixed green card conversion system (CORRECTED SPEC-8):")
    print(f"  Annual conversions: {annual_cap} (FIXED - computed once)")
    print(f"  Residual fraction: {residual:.6f}")
    print(f"  Based on: {format_number(GREEN_CARD_CAP_ABS)} / {format_number(REAL_US_WORKFORCE_SIZE)} * {format_number(config.initial_workers)}")
    print(f"  Cap proportion: {format_percentage(cap_proportion, 6)}")
    
    # Per-country cap information
    if config.country_cap_enabled:
        from .empirical_params import calculate_per_country_cap
        per_country_cap, per_country_residual = calculate_per_country_cap(annual_cap)
        print(f"  Per-country cap: ENABLED ({format_percentage(PER_COUNTRY_CAP_SHARE)})")
        print(f"  Max per country: {per_country_cap} conversions/year (FIXED)")
        print(f"  Per-country residual: {per_country_residual:.6f}")
        print(f"  Queue mode: Separate nationality queues with FIFO within each")
    else:
        print(f"  Per-country cap: DISABLED")
        print(f"  Queue mode: Single global FIFO queue")
    
    # Enhanced wage differentiation info (CORRECTED FOR SPEC-8)
    temp_job_change_prob = JOB_CHANGE_PROB_PERM * (1 - TEMP_JOB_CHANGE_PENALTY)
    print(f"\nEnhanced wage differentiation (CORRECTED SPEC-8):")
    print(f"  Starting wage: {format_currency(STARTING_WAGE)}")
    print(f"  Job change probability (permanent): {format_percentage(JOB_CHANGE_PROB_PERM)}")
    print(f"  Job change probability (temporary): {format_percentage(temp_job_change_prob)}")
    print(f"  Wage jump on job change (permanent): {format_percentage(WAGE_JUMP_FACTOR_MEAN_PERM - 1)}")
    print(f"  Wage jump on job change (temporary): {format_percentage(WAGE_JUMP_FACTOR_MEAN_TEMP - 1)}")
    print(f"  Converted workers switch to permanent parameters: NEXT YEAR after conversion")
    
    # Nationality distribution
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
    
    # Backlog analysis information
    if config.compare_backlogs:
        print(f"\nBacklog analysis: ENABLED")
        print(f"  Will compare final-year backlogs by nationality")
        print(f"  Expected: Total backlog identical across scenarios, distribution differs")
    
    if live_data:
        print("\nUsing live-fetched data:")
        print(f"  H-1B share: {format_percentage(live_data.get('h1b_share', H1B_SHARE))}")
        print(f"  Annual H-1B entry rate: {format_percentage(live_data.get('annual_h1b_entry_rate', ANNUAL_H1B_ENTRY_RATE))}")
        if 'green_card_cap' in live_data:
            print(f"  Green card cap: {format_number(live_data['green_card_cap'])}")
        if 'it_median_wage' in live_data:
            print(f"  IT sector median wage: {format_currency(live_data['it_median_wage'])}")
    else:
        print("\nUsing default empirical parameters:")
        print(f"  H-1B share: {format_percentage(H1B_SHARE)}")
        print(f"  Annual permanent entry rate: {format_percentage(ANNUAL_PERMANENT_ENTRY_RATE)}")
        print(f"  Annual H-1B entry rate: {format_percentage(ANNUAL_H1B_ENTRY_RATE)}")
    
    print("="*60)

def main() -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.quiet, args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Check Python version
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        required_version = PYTHON_MIN_VERSION
        if python_version < required_version:
            print(f"Error: Python {required_version}+ required, found {python_version}")
            return 1
        
        # Determine country cap setting
        if args.country_cap:
            country_cap_enabled = True
        elif args.no_country_cap:
            country_cap_enabled = False
        else:
            country_cap_enabled = ENABLE_COUNTRY_CAP
        
        # Check visualization requirements
        if (args.visualize_results or args.compare_backlogs) and not VISUALIZATION_AVAILABLE:
            print("Error: Visualization libraries not available. Install with:")
            print("pip install matplotlib seaborn plotly pandas")
            return 1
        
        # Create base configuration
        base_config = SimulationConfig(
            initial_workers=args.initial_workers,
            years=args.years,
            seed=args.seed,
            live_fetch=args.live_fetch,
            output_path=args.output,
            agent_mode=not args.count_mode,
            show_nationality_summary=args.show_nationality_summary,
            country_cap_enabled=country_cap_enabled,
            compare_backlogs=args.compare_backlogs,
            debug=args.debug  # CORRECTED FOR SPEC-8
        )
        
        # Validate configuration
        errors = validate_configuration(base_config)
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  • {error}")
            return 1
        
        # Fetch live data if requested
        live_data = None
        updated_nationality_distribution = None
        if base_config.live_fetch:
            if not args.quiet:
                print("Fetching live workforce, wage, nationality, and backlog data...")
            live_data = fetch_live_data()
            if live_data:
                if not args.quiet:
                    print("Successfully fetched live data")
                
                updated_nationality_distribution = update_nationality_distribution(live_data)
                if updated_nationality_distribution != TEMP_NATIONALITY_DISTRIBUTION:
                    print("Updated nationality distribution from live data")
            else:
                print("Warning: Failed to fetch live data, using defaults")
        
        # Handle different modes
        if args.compare_backlogs:
            # Backlog analysis mode
            if not args.quiet:
                print_simulation_header(base_config, live_data, updated_nationality_distribution)
                print("\n📊 BACKLOG ANALYSIS MODE: Running both scenarios for backlog comparison...")
            
            # Run uncapped scenario
            uncapped_config = SimulationConfig(
                initial_workers=args.initial_workers,
                years=args.years,
                seed=args.seed,
                live_fetch=False,
                output_path=f"{args.output_dir}/uncapped_results.csv",
                agent_mode=not args.count_mode,
                show_nationality_summary=False,
                country_cap_enabled=False,
                compare_backlogs=True,
                debug=args.debug
            )
            
            results_uncapped, backlog_uncapped = run_simulation_scenario(
                uncapped_config, "Uncapped", live_data, 
                updated_nationality_distribution, args.quiet
            )
            
            # Run capped scenario
            capped_config = SimulationConfig(
                initial_workers=args.initial_workers,
                years=args.years,
                seed=args.seed,
                live_fetch=False,
                output_path=f"{args.output_dir}/capped_results.csv",
                agent_mode=not args.count_mode,
                show_nationality_summary=False,
                country_cap_enabled=True,
                compare_backlogs=True,
                debug=args.debug
            )
            
            results_capped, backlog_capped = run_simulation_scenario(
                capped_config, "Capped", live_data, 
                updated_nationality_distribution, args.quiet
            )
            
            # Save backlog analyses to CSV
            backlog_uncapped_path = Path(args.output_dir) / "backlog_uncapped.csv"
            backlog_capped_path = Path(args.output_dir) / "backlog_capped.csv"
            
            save_backlog_analysis(backlog_uncapped, str(backlog_uncapped_path))
            save_backlog_analysis(backlog_capped, str(backlog_capped_path))
            
            # Generate visualizations
            if not args.quiet:
                print("\n📊 Generating backlog comparison visualizations...")
            
            if not validate_dataframes(results_uncapped, results_capped):
                print("Error: Invalid data for wage visualization")
                return 1
                
            backlog_df_uncapped = backlog_uncapped.to_dataframe()
            backlog_df_capped = backlog_capped.to_dataframe()
            
            if not validate_backlog_dataframes(backlog_df_uncapped, backlog_df_capped):
                print("Error: Invalid data for backlog visualization")
                return 1
            
            # Create visualizer and generate charts
            visualizer = SimulationVisualizer(
                output_dir=args.output_dir,
                save_plots=not args.skip_plots
            )
            
            # Generate all visualizations (wage + backlog)
            wage_files = visualizer.generate_all_visualizations(
                results_uncapped, results_capped
            )
            
            backlog_files = visualizer.generate_backlog_visualizations(
                backlog_df_uncapped, backlog_df_capped
            )
            
            # Combine all generated files
            generated_files = {**wage_files, **backlog_files}
            
            # Print results
            if not args.quiet:
                print("\n✅ Backlog analysis complete!")
                print("Generated files:")
                for viz_name, filepath in generated_files.items():
                    if filepath:
                        print(f"  📈 {viz_name}: {filepath}")
                
                print(f"\nCSV files:")
                print(f"  📄 backlog_uncapped.csv: {backlog_uncapped_path}")
                print(f"  📄 backlog_capped.csv: {backlog_capped_path}")
                
                # CORRECTED FOR SPEC-8: Show invariant validation
                print(f"\nCORRECTED SPEC-8 Invariant Validation:")
                total_backlog_diff = abs(backlog_uncapped.total_backlog - backlog_capped.total_backlog)
                print(f"  Total backlog difference: {total_backlog_diff} (should be ≤ 1)")
                print(f"  Total backlog (no cap): {backlog_uncapped.total_backlog:,} workers")
                print(f"  Total backlog (7% cap): {backlog_capped.total_backlog:,} workers")
                print(f"  Invariant C {'✅ PASSED' if total_backlog_diff <= 1 else '❌ FAILED'}")
        
        elif args.visualize_results:
            # Standard visualization mode
            # Print header once
            if not args.quiet:
                print_simulation_header(base_config, live_data, updated_nationality_distribution)
                print("\n🎨 VISUALIZATION MODE: Running both scenarios for comparison...")
            
            # Run both scenarios (similar to backlog analysis but focus on visualization)
            uncapped_config = SimulationConfig(
                initial_workers=args.initial_workers,
                years=args.years,
                seed=args.seed,
                live_fetch=False,
                output_path=f"{args.output_dir}/uncapped_results.csv",
                agent_mode=not args.count_mode,
                show_nationality_summary=False,
                country_cap_enabled=False,
                debug=args.debug
            )
            
            results_uncapped, _ = run_simulation_scenario(
                uncapped_config, "Uncapped", live_data, 
                updated_nationality_distribution, args.quiet
            )
            
            capped_config = SimulationConfig(
                initial_workers=args.initial_workers,
                years=args.years,
                seed=args.seed,
                live_fetch=False,
                output_path=f"{args.output_dir}/capped_results.csv",
                agent_mode=not args.count_mode,
                show_nationality_summary=False,
                country_cap_enabled=True,
                debug=args.debug
            )
            
            results_capped, _ = run_simulation_scenario(
                capped_config, "Capped", live_data, 
                updated_nationality_distribution, args.quiet
            )
            
            # Generate visualizations
            if not args.quiet:
                print("\n📊 Generating comparative visualizations...")
            
            if not validate_dataframes(results_uncapped, results_capped):
                print("Error: Invalid data for visualization")
                return 1
            
            visualizer = SimulationVisualizer(
                output_dir=args.output_dir,
                save_plots=not args.skip_plots
            )
            
            generated_files = visualizer.generate_all_visualizations(
                results_uncapped, results_capped
            )
            
            # Print results
            if not args.quiet:
                print("\n✅ Visualization complete!")
                print("Generated files:")
                for viz_name, filepath in generated_files.items():
                    if filepath:
                        print(f"  📈 {viz_name}: {filepath}")
                
                # CORRECTED FOR SPEC-8: Show conversion analysis
                print(f"\nCORRECTED SPEC-8 Conversion Analysis:")
                conversions_uncapped = results_uncapped['converted_temps'].tolist()[1:6]  # First 5 years
                conversions_capped = results_capped['converted_temps'].tolist()[1:6]
                print(f"  Conversions per year (uncapped): {conversions_uncapped}")
                print(f"  Conversions per year (capped):   {conversions_capped}")
                
                # Check if conversions are flat
                if conversions_uncapped:
                    flat_check_uncapped = all(abs(x - conversions_uncapped[0]) <= 1 for x in conversions_uncapped)
                    flat_check_capped = all(abs(x - conversions_capped[0]) <= 1 for x in conversions_capped)
                    print(f"  Flat conversions check: {'✅ PASSED' if flat_check_uncapped and flat_check_capped else '❌ FAILED'}")
        
        else:
            # Standard single-scenario mode
            if not args.quiet:
                print_simulation_header(base_config, live_data, updated_nationality_distribution)
            
            # Run simulation
            simulation = Simulation(base_config)
            
            # Update nationality distribution in simulation if live data was fetched
            if updated_nationality_distribution:
                simulation.temp_nationality_distribution = updated_nationality_distribution
                simulation._validate_nationality_distribution()
            
            states = simulation.run()
            
            # Save results
            save_simulation_results(states, base_config.output_path, include_nationality_columns=True)
            
            # Export nationality report if requested
            if args.export_nationality_report:
                if base_config.agent_mode:
                    workers = simulation.to_agent_model()
                    export_nationality_report(workers, args.export_nationality_report)
                    if not args.quiet:
                        print(f"Nationality report exported to: {args.export_nationality_report}")
                else:
                    print("Warning: Nationality report export requires agent-mode")
            
            # Print results (unless quiet mode)
            if not args.quiet:
                from .cli import print_simulation_results
                print_simulation_results(simulation)
                print_data_sources(live_data)
                
                # CORRECTED FOR SPEC-8: Show validation results
                if simulation.validate_fixed_conversion_invariants():
                    print(f"\n✅ CORRECTED SPEC-8 Invariants: PASSED")
                else:
                    print(f"\n❌ CORRECTED SPEC-8 Invariants: FAILED")
        
        if not args.quiet:
            print(f"\n✅ Simulation completed successfully!")
            if not (args.visualize_results or args.compare_backlogs):
                print(f"Results saved to: {base_config.output_path}")
            print(f"\nNext steps:")
            print(f"  • Examine flat conversion plots (should show constant annual conversions)")
            print(f"  • Verify total backlog equality between scenarios")
            print(f"  • Compare wage differentiation (permanent vs temporary workers)")
            if not args.compare_backlogs:
                print(f"  • Use --compare-backlogs to verify SPEC-8 invariants")
            print(f"  • Use --debug to see detailed conversion cap information")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        if args.verbose or args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
