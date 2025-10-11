# src/simulation/cli.py
"""
Command-line interface for the workforce simulation.
CLEANUP: Streamlined to work with cleaned up models and removed dependencies.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .models import SimulationConfig, BacklogAnalysis
from .sim import Simulation

logger = logging.getLogger(__name__)

def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s:%(name)s:%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def create_config_from_args(args) -> SimulationConfig:
    """Create SimulationConfig from command line arguments."""
    return SimulationConfig(
        initial_workers=args.initial_workers,
        years=args.years,
        seed=args.seed,
        output_path=args.output,
        country_cap_enabled=args.country_cap,
        compare_backlogs=args.compare,
        debug=args.debug,
        start_year=args.start_year
    )

def save_simulation_results_csv(states, filepath: str):
    """Save simulation results to CSV file."""
    import csv
    from pathlib import Path
    
    # Ensure output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            'year', 'total_workers', 'permanent_workers', 'temporary_workers',
            'new_permanent', 'new_temporary', 'converted_temps',
            'avg_wage_total', 'avg_wage_permanent', 'avg_wage_temporary',
            'total_wage_bill', 'h1b_share', 'permanent_share',
            'annual_conversion_cap', 'cumulative_conversions'
        ])
        
        # Write data rows
        for state in states:
            writer.writerow([
                state.year, state.total_workers, state.permanent_workers, state.temporary_workers,
                state.new_permanent, state.new_temporary, state.converted_temps,
                state.avg_wage_total, state.avg_wage_permanent, state.avg_wage_temporary,
                state.total_wage_bill, state.h1b_share, state.permanent_share,
                state.annual_conversion_cap, state.cumulative_conversions
            ])

def run_single_simulation(config: SimulationConfig) -> None:
    """Run a single simulation with the given configuration."""
    logger.info(f"Running simulation: {config.initial_workers:,} workers, {config.years} years")
    
    # Create and run simulation
    sim = Simulation(config)
    states = sim.run()
    
    # Save results
    try:
        save_simulation_results_csv(states, config.output_path)
        logger.info(f"Results saved to: {config.output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def run_comparative_analysis(config: SimulationConfig) -> None:
    """Run comparative analysis between capped and uncapped scenarios."""
    logger.info("Running comparative analysis (capped vs uncapped)")
    
    # Run uncapped simulation
    logger.info("Running uncapped simulation...")
    config_uncapped = SimulationConfig(
        initial_workers=config.initial_workers,
        years=config.years,
        seed=config.seed,
        country_cap_enabled=False,
        debug=config.debug,
        start_year=config.start_year
    )
    
    sim_uncapped = Simulation(config_uncapped)
    states_uncapped = sim_uncapped.run()
    
    # Run capped simulation
    logger.info("Running capped simulation...")
    config_capped = SimulationConfig(
        initial_workers=config.initial_workers,
        years=config.years,
        seed=config.seed,
        country_cap_enabled=True,
        debug=config.debug,
        start_year=config.start_year
    )
    
    sim_capped = Simulation(config_capped)
    states_capped = sim_capped.run()
    
    # Save individual results
    uncapped_path = config.output_path.replace('.csv', '_uncapped.csv')
    capped_path = config.output_path.replace('.csv', '_capped.csv')
    
    try:
        save_simulation_results_csv(states_uncapped, uncapped_path)
        save_simulation_results_csv(states_capped, capped_path)
        logger.info(f"Uncapped results saved to: {uncapped_path}")
        logger.info(f"Capped results saved to: {capped_path}")
    except Exception as e:
        logger.error(f"Error saving simulation results: {e}")
        raise
    
    # Generate comparative analysis
    try:
        backlog_uncapped = BacklogAnalysis.from_simulation(sim_uncapped, "uncapped")
        backlog_capped = BacklogAnalysis.from_simulation(sim_capped, "capped")
        
        # Save backlog analyses
        backlog_uncapped.save_csv(config.output_path.replace('.csv', '_backlog_uncapped.csv'))
        backlog_capped.save_csv(config.output_path.replace('.csv', '_backlog_capped.csv'))
        
        # Generate visualizations if available
        try:
            generate_comparison_charts(states_uncapped, states_capped, backlog_uncapped, backlog_capped)
        except ImportError:
            logger.warning("Visualization libraries not available. Install matplotlib and seaborn for charts.")
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
        
        # Print comparison summary
        print_comparison_summary(states_uncapped, states_capped, backlog_uncapped, backlog_capped)
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {e}")
        raise

def generate_comparison_charts(states_uncapped, states_capped, backlog_uncapped, backlog_capped):
    """Generate comparison charts if visualization libraries are available."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Set style
    plt.style.use('default')
    sns.set_palette("muted")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Immigration Policy Comparison: Capped vs Uncapped', fontsize=16, fontweight='bold')
    
    # 1. Wage growth comparison
    years_uncapped = [state.year for state in states_uncapped]
    wages_uncapped = [state.avg_wage_total for state in states_uncapped]
    years_capped = [state.year for state in states_capped]
    wages_capped = [state.avg_wage_total for state in states_capped]
    
    axes[0, 0].plot(years_uncapped, wages_uncapped, label='No Per-Country Cap', linewidth=2, color='#1f77b4')
    axes[0, 0].plot(years_capped, wages_capped, label='7% Per-Country Cap', linewidth=2, color='#ff7f0e')
    axes[0, 0].set_title('Average Worker Wage Comparison Over Time')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Average Wage ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Green card conversions per year
    conversions_uncapped = [state.converted_temps for state in states_uncapped[1:]]  # Skip first year
    conversions_capped = [state.converted_temps for state in states_capped[1:]]
    conversion_years = years_uncapped[1:]
    
    width = 0.35
    x = range(len(conversion_years))
    axes[0, 1].bar([i - width/2 for i in x], conversions_uncapped, width, label='No Per-Country Cap', alpha=0.8, color='#1f77b4')
    axes[0, 1].bar([i + width/2 for i in x], conversions_capped, width, label='7% Per-Country Cap', alpha=0.8, color='#ff7f0e')
    axes[0, 1].set_title('Green Card Conversions Per Year')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Number of Conversions')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(year) for year in conversion_years], rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Final-Year Green Card Backlog by Nationality
    backlog_df_uncapped = backlog_uncapped.to_dataframe()
    backlog_df_capped = backlog_capped.to_dataframe()
    
    # Get top nationalities by backlog size
    combined_backlogs = {}
    for _, row in backlog_df_uncapped.iterrows():
        combined_backlogs[row['nationality']] = row['backlog_size']
    for _, row in backlog_df_capped.iterrows():
        if row['nationality'] in combined_backlogs:
            combined_backlogs[row['nationality']] += row['backlog_size']
        else:
            combined_backlogs[row['nationality']] = row['backlog_size']
    
    top_nationalities = sorted(combined_backlogs.items(), key=lambda x: x[1], reverse=True)[:10]
    top_countries = [x[0] for x in top_nationalities]
    
    uncapped_values = []
    capped_values = []
    for country in top_countries:
        uncapped_val = backlog_df_uncapped[backlog_df_uncapped['nationality'] == country]['backlog_size'].iloc[0] if len(backlog_df_uncapped[backlog_df_uncapped['nationality'] == country]) > 0 else 0
        capped_val = backlog_df_capped[backlog_df_capped['nationality'] == country]['backlog_size'].iloc[0] if len(backlog_df_capped[backlog_df_capped['nationality'] == country]) > 0 else 0
        uncapped_values.append(uncapped_val)
        capped_values.append(capped_val)
    
    x = range(len(top_countries))
    axes[1, 0].bar([i - width/2 for i in x], uncapped_values, width, label='No Per-Country Cap', alpha=0.8, color='#1f77b4')
    axes[1, 0].bar([i + width/2 for i in x], capped_values, width, label='7% Per-Country Cap', alpha=0.8, color='#ff7f0e')
    axes[1, 0].set_title('Final-Year Green Card Backlog by Nationality')
    axes[1, 0].set_xlabel('Nationality')
    axes[1, 0].set_ylabel('Backlog Size (# of Temporary Workers)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(top_countries, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. H-1B share evolution
    h1b_uncapped = [state.h1b_share for state in states_uncapped]
    h1b_capped = [state.h1b_share for state in states_capped]
    
    axes[1, 1].plot(years_uncapped, h1b_uncapped, label='No Per-Country Cap', linewidth=2, color='#1f77b4')
    axes[1, 1].plot(years_capped, h1b_capped, label='7% Per-Country Cap', linewidth=2, color='#ff7f0e')
    axes[1, 1].set_title('H-1B Share of Workforce Over Time')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('H-1B Share (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path('output')
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / 'immigration_policy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Comparison chart saved to: {output_path / 'immigration_policy_comparison.png'}")

def print_comparison_summary(states_uncapped, states_capped, backlog_uncapped, backlog_capped):
    """Print comparison summary."""
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("="*60)
    
    # Final states
    final_uncapped = states_uncapped[-1]
    final_capped = states_capped[-1]
    
    print(f"Final workforce size:")
    print(f"  Uncapped: {final_uncapped.total_workers:,}")
    print(f"  Capped:   {final_capped.total_workers:,}")
    print(f"  Difference: {final_capped.total_workers - final_uncapped.total_workers:,}")
    
    print(f"\nFinal average wages:")
    print(f"  Uncapped: ${final_uncapped.avg_wage_total:,.0f}")
    print(f"  Capped:   ${final_capped.avg_wage_total:,.0f}")
    print(f"  Difference: ${final_capped.avg_wage_total - final_uncapped.avg_wage_total:,.0f}")
    
    print(f"\nTotal green card backlogs:")
    print(f"  Uncapped: {backlog_uncapped.total_backlog:,}")
    print(f"  Capped:   {backlog_capped.total_backlog:,}")
    print(f"  Difference: {backlog_capped.total_backlog - backlog_uncapped.total_backlog:,}")
    
    # Backlog validation
    backlog_diff = abs(backlog_uncapped.total_backlog - backlog_capped.total_backlog)
    print(f"\nSPEC-10 Backlog Validation:")
    print(f"  Total backlog difference: {backlog_diff} (should be ≤ 1)")
    print(f"  Backlog invariant {'✅ PASSED' if backlog_diff <= 1 else '❌ FAILED'}")
    
    # Top backlogs by nationality
    print(f"\nTop 3 backlogs by nationality (Uncapped):")
    top_uncapped = backlog_uncapped.get_top_backlogs(3)
    for nationality, size in top_uncapped.items():
        print(f"  {nationality}: {size:,}")
    
    print(f"\nTop 3 backlogs by nationality (Capped):")
    top_capped = backlog_capped.get_top_backlogs(3)
    for nationality, size in top_capped.items():
        print(f"  {nationality}: {size:,}")
    
    print("="*60)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Workforce Growth Simulation - Immigration Policy Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation
  python -m src.simulation --initial-workers 50000 --years 10

  # Policy comparison  
  python -m src.simulation --initial-workers 100000 --years 20 --compare

  # Enable per-country caps with debug output
  python -m src.simulation --initial-workers 75000 --country-cap --debug

  # Custom output location
  python -m src.simulation --initial-workers 25000 --output results/my_simulation.csv
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--initial-workers', 
        type=int, 
        required=True,
        help='Initial workforce size (required)'
    )
    
    parser.add_argument(
        '--years', 
        type=int, 
        default=20,
        help='Number of years to simulate (default: 20)'
    )
    
    # Policy options
    parser.add_argument(
        '--country-cap',
        action='store_true',
        help='Enable per-country caps (7%% limitation)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true', 
        help='Compare capped vs uncapped scenarios'
    )
    
    # Configuration options
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=2025,
        help='Starting year for simulation (default: 2025)'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='data/simulation_results.csv',
        help='Output file path (default: data/simulation_results.csv)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    try:
        config = create_config_from_args(args)
    except Exception as e:
        logger.error(f"Error creating configuration: {e}")
        sys.exit(1)
    
    # Print simulation info
    print("\n" + "="*60)
    print("WORKFORCE GROWTH SIMULATION")
    print("="*60)
    print(f"Initial workforce: {config.initial_workers:,}")
    print(f"Simulation years: {config.years}")
    print(f"Per-country cap: {'ENABLED' if config.country_cap_enabled else 'DISABLED'}")
    print(f"Random seed: {config.seed or 'None (random)'}")
    print("="*60)
    
    # Run simulation(s)
    try:
        if args.compare:
            run_comparative_analysis(config)
        else:
            run_single_simulation(config)
        
        logger.info("Simulation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
