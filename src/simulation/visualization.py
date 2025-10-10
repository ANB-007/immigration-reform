# src/simulation/visualization.py
"""
Visualization module for workforce growth simulation.
Provides comparative analysis charts and interactive visualizations.
Updated for SPEC-7: Comparative backlog analysis by nationality.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import math
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive charts will be disabled.")

from .empirical_params import OUTPUT_DIR, PLOT_DPI, PLOT_STYLE, PLOT_PALETTE, SAVE_PLOTS

# Configure logging and plotting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up matplotlib and seaborn styling
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_theme(style=PLOT_STYLE, palette=PLOT_PALETTE)

# Color scheme for consistent branding
COLORS = {
    'uncapped': '#2E86AB',      # Blue
    'capped': '#A23B72',        # Purple/Red
    'permanent': '#F18F01',     # Orange
    'temporary': '#C73E1D',     # Red
    'accent': '#FFE066',        # Yellow
    'neutral': '#6C757D'        # Gray
}

class SimulationVisualizer:
    """
    Visualization class for workforce simulation results.
    Provides methods for creating comparative charts and analysis plots.
    Updated for SPEC-7 to include backlog analysis visualizations.
    """
    
    def __init__(self, output_dir: str = OUTPUT_DIR, save_plots: bool = SAVE_PLOTS):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
            save_plots: Whether to automatically save plots
        """
        self.output_dir = Path(output_dir)
        self.save_plots = save_plots
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SimulationVisualizer initialized. Output directory: {self.output_dir}")
    
    def compare_average_wages(self, results_uncapped: pd.DataFrame, 
                            results_capped: pd.DataFrame) -> str:
        """
        Create a line chart comparing average wages over time between scenarios.
        
        Args:
            results_uncapped: DataFrame with uncapped simulation results
            results_capped: DataFrame with capped simulation results
            
        Returns:
            Path to saved plot file
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot wage trajectories
        ax.plot(results_uncapped['year'], results_uncapped['avg_wage_total'], 
                label='No Per-Country Cap', color=COLORS['uncapped'], 
                linewidth=3, marker='o', markersize=4)
        
        ax.plot(results_capped['year'], results_capped['avg_wage_total'], 
                label='7% Per-Country Cap', color=COLORS['capped'], 
                linewidth=3, marker='s', markersize=4)
        
        # Formatting
        ax.set_title('Average Worker Wage Comparison Over Time', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Wage (USD)', fontsize=12, fontweight='bold')
        
        # Format y-axis as currency
        formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
        ax.yaxis.set_major_formatter(formatter)
        
        # Legend and grid
        ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot
        if self.save_plots:
            filename = self.output_dir / "wage_comparison_over_time.png"
            plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
            logger.info(f"✅ Saved plot: {filename}")
            
        plt.show()
        return str(filename) if self.save_plots else ""
    
    def plot_final_wage_comparison(self, results_uncapped: pd.DataFrame, 
                                 results_capped: pd.DataFrame) -> str:
        """
        Create a bar chart comparing final-year average wages.
        
        Args:
            results_uncapped: DataFrame with uncapped simulation results
            results_capped: DataFrame with capped simulation results
            
        Returns:
            Path to saved plot file
        """
        # Get final year wages
        final_uncapped = results_uncapped.iloc[-1]['avg_wage_total']
        final_capped = results_capped.iloc[-1]['avg_wage_total']
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Scenario': ['No Per-Country Cap', '7% Per-Country Cap'],
            'Average Final Wage': [final_uncapped, final_capped],
            'Colors': [COLORS['uncapped'], COLORS['capped']]
        })
        
        if PLOTLY_AVAILABLE:
            # Create interactive Plotly bar chart
            fig = px.bar(comparison_df, x='Scenario', y='Average Final Wage', 
                        color='Scenario',
                        color_discrete_sequence=[COLORS['uncapped'], COLORS['capped']],
                        title='Final-Year Average Wage Comparison',
                        text='Average Final Wage')
            
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(
                yaxis_title='Average Wage (USD)', 
                xaxis_title='Scenario', 
                showlegend=False,
                title_x=0.5,
                font=dict(size=12),
                height=600,
                width=800
            )
            
            # Save interactive chart
            if self.save_plots:
                filename = self.output_dir / "final_wage_comparison.html"
                fig.write_html(str(filename))
                logger.info(f"✅ Saved interactive chart: {filename}")
                
            fig.show()
            return str(filename) if self.save_plots else ""
        
        else:
            # Fallback to matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.bar(comparison_df['Scenario'], comparison_df['Average Final Wage'],
                         color=[COLORS['uncapped'], COLORS['capped']], 
                         alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add value labels on bars
            for bar, value in zip(bars, comparison_df['Average Final Wage']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'${value:,.0f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=11)
            
            ax.set_title('Final-Year Average Wage Comparison', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Average Wage (USD)', fontsize=12, fontweight='bold')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            plt.tight_layout()
            
            if self.save_plots:
                filename = self.output_dir / "final_wage_comparison.png"
                plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
                logger.info(f"✅ Saved plot: {filename}")
                
            plt.show()
            return str(filename) if self.save_plots else ""
    
    def plot_workforce_composition(self, results_uncapped: pd.DataFrame, 
                                 results_capped: pd.DataFrame) -> str:
        """
        Create stacked area chart showing workforce composition over time.
        
        Args:
            results_uncapped: DataFrame with uncapped simulation results
            results_capped: DataFrame with capped simulation results
            
        Returns:
            Path to saved plot file
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot uncapped scenario
        ax1.fill_between(results_uncapped['year'], 0, results_uncapped['permanent_workers'],
                        color=COLORS['permanent'], alpha=0.7, label='Permanent Workers')
        ax1.fill_between(results_uncapped['year'], results_uncapped['permanent_workers'],
                        results_uncapped['total_workers'],
                        color=COLORS['temporary'], alpha=0.7, label='Temporary Workers')
        
        ax1.set_title('Workforce Composition\n(No Per-Country Cap)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Workers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot capped scenario
        ax2.fill_between(results_capped['year'], 0, results_capped['permanent_workers'],
                        color=COLORS['permanent'], alpha=0.7, label='Permanent Workers')
        ax2.fill_between(results_capped['year'], results_capped['permanent_workers'],
                        results_capped['total_workers'],
                        color=COLORS['temporary'], alpha=0.7, label='Temporary Workers')
        
        ax2.set_title('Workforce Composition\n(7% Per-Country Cap)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Number of Workers')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = self.output_dir / "workforce_composition_comparison.png"
            plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
            logger.info(f"✅ Saved plot: {filename}")
            
        plt.show()
        return str(filename) if self.save_plots else ""
    
    def plot_conversion_statistics(self, results_capped: pd.DataFrame) -> str:
        """
        Create visualization of conversion statistics for capped scenario.
        
        Args:
            results_capped: DataFrame with capped simulation results
            
        Returns:
            Path to saved plot file
        """
        if 'converted_temps' not in results_capped.columns:
            logger.warning("No conversion data available for visualization")
            return ""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Conversions over time
        ax1.bar(results_capped['year'], results_capped['converted_temps'],
               color=COLORS['accent'], alpha=0.8, edgecolor=COLORS['capped'])
        ax1.set_title('Green Card Conversions Per Year (7% Cap)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Conversions')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative conversions
        cumulative_conversions = results_capped['converted_temps'].cumsum()
        ax2.plot(results_capped['year'], cumulative_conversions,
                color=COLORS['capped'], linewidth=3, marker='o')
        ax2.fill_between(results_capped['year'], 0, cumulative_conversions,
                        color=COLORS['capped'], alpha=0.3)
        ax2.set_title('Cumulative Green Card Conversions', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Cumulative Conversions')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = self.output_dir / "conversion_statistics.png"
            plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
            logger.info(f"✅ Saved plot: {filename}")
            
        plt.show()
        return str(filename) if self.save_plots else ""
    
    def compare_backlog_sizes(self, backlog_uncapped: pd.DataFrame, 
                            backlog_capped: pd.DataFrame) -> str:
        """
        Create a bar chart comparing final-year backlogs by nationality (NEW FOR SPEC-7).
        
        Args:
            backlog_uncapped: DataFrame with uncapped backlog data
            backlog_capped: DataFrame with capped backlog data
            
        Returns:
            Path to saved plot file
        """
        # Merge the data on nationality
        merged = backlog_uncapped.merge(backlog_capped, on='nationality', suffixes=('_uncapped', '_capped'))
        merged = merged.sort_values('backlog_size_capped', ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create bar positions
        x = np.arange(len(merged))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, merged['backlog_size_uncapped'], width, 
                      label='No Per-Country Cap', color=COLORS['uncapped'], alpha=0.8)
        bars2 = ax.bar(x + width/2, merged['backlog_size_capped'], width,
                      label='7% Per-Country Cap', color=COLORS['capped'], alpha=0.8)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only add labels for non-zero bars
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        # Formatting
        ax.set_title('Final-Year Green Card Backlog by Nationality', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Nationality', fontsize=12, fontweight='bold')
        ax.set_ylabel('Backlog Size (# of Temporary Workers)', fontsize=12, fontweight='bold')
        
        # Format y-axis with commas
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Set x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(merged['nationality'], rotation=45, ha='right')
        
        # Legend and grid
        ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        if self.save_plots:
            filename = self.output_dir / "backlog_comparison.png"
            plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
            logger.info(f"✅ Saved backlog comparison chart: {filename}")
            
        plt.show()
        return str(filename) if self.save_plots else ""
    
    def backlog_bar_interactive(self, backlog_uncapped: pd.DataFrame, 
                              backlog_capped: pd.DataFrame) -> str:
        """
        Create an interactive backlog comparison using Plotly (NEW FOR SPEC-7).
        
        Args:
            backlog_uncapped: DataFrame with uncapped backlog data
            backlog_capped: DataFrame with capped backlog data
            
        Returns:
            Path to saved interactive chart file
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create interactive backlog chart.")
            return ""
        
        # Combine data for plotting
        df = pd.concat([
            backlog_uncapped.assign(Scenario='No Per-Country Cap'), 
            backlog_capped.assign(Scenario='7% Per-Country Cap')
        ])
        
        # Sort by capped backlog size for consistent ordering
        capped_order = backlog_capped.sort_values('backlog_size', ascending=False)['nationality'].tolist()
        df['nationality'] = pd.Categorical(df['nationality'], categories=capped_order, ordered=True)
        df = df.sort_values('nationality')
        
        # Create interactive bar chart
        fig = px.bar(df, x='nationality', y='backlog_size', color='Scenario', 
                    barmode='group',
                    color_discrete_sequence=[COLORS['uncapped'], COLORS['capped']],
                    title='Final-Year Backlog Comparison by Nationality',
                    text='backlog_size')
        
        # Update text format
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        
        # Update layout
        fig.update_layout(
            xaxis_title='Nationality', 
            yaxis_title='Backlog Size (# of Workers)',
            legend_title='Scenario',
            title_x=0.5,
            font=dict(size=12),
            height=700,
            width=1000,
            xaxis={'categoryorder': 'array', 'categoryarray': capped_order}
        )
        
        # Format y-axis
        fig.update_yaxis(tickformat=',.')
        
        # Save interactive chart
        if self.save_plots:
            filename = self.output_dir / "backlog_comparison.html"
            fig.write_html(str(filename))
            logger.info(f"✅ Saved interactive backlog chart: {filename}")
            
        fig.show()
        return str(filename) if self.save_plots else ""
    
    def create_summary_dashboard(self, results_uncapped: pd.DataFrame, 
                               results_capped: pd.DataFrame) -> str:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            results_uncapped: DataFrame with uncapped simulation results
            results_capped: DataFrame with capped simulation results
            
        Returns:
            Path to saved dashboard file
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create interactive dashboard.")
            return ""
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Average Wage Over Time',
                'Final Wage Comparison', 
                'Total Workers Over Time',
                'Green Card Conversions (Capped Only)'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Wage comparison
        fig.add_trace(
            go.Scatter(x=results_uncapped['year'], y=results_uncapped['avg_wage_total'],
                      mode='lines+markers', name='No Cap', 
                      line=dict(color=COLORS['uncapped'], width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=results_capped['year'], y=results_capped['avg_wage_total'],
                      mode='lines+markers', name='7% Cap',
                      line=dict(color=COLORS['capped'], width=3)),
            row=1, col=1
        )
        
        # Plot 2: Final wage bars
        final_wages = [results_uncapped.iloc[-1]['avg_wage_total'], 
                      results_capped.iloc[-1]['avg_wage_total']]
        fig.add_trace(
            go.Bar(x=['No Cap', '7% Cap'], y=final_wages,
                  marker_color=[COLORS['uncapped'], COLORS['capped']],
                  name='Final Wages', showlegend=False),
            row=1, col=2
        )
        
        # Plot 3: Total workers
        fig.add_trace(
            go.Scatter(x=results_uncapped['year'], y=results_uncapped['total_workers'],
                      mode='lines', name='Total (No Cap)', 
                      line=dict(color=COLORS['uncapped'], width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=results_capped['year'], y=results_capped['total_workers'],
                      mode='lines', name='Total (7% Cap)',
                      line=dict(color=COLORS['capped'], width=2)),
            row=2, col=1
        )
        
        # Plot 4: Conversions (if available)
        if 'converted_temps' in results_capped.columns:
            fig.add_trace(
                go.Bar(x=results_capped['year'], y=results_capped['converted_temps'],
                      marker_color=COLORS['accent'], name='Conversions', showlegend=False),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Workforce Simulation Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True,
            font=dict(size=10)
        )
        
        # Save dashboard
        if self.save_plots:
            filename = self.output_dir / "simulation_dashboard.html"
            fig.write_html(str(filename))
            logger.info(f"✅ Saved dashboard: {filename}")
            
        fig.show()
        return str(filename) if self.save_plots else ""
    
    def generate_all_visualizations(self, results_uncapped: pd.DataFrame, 
                                  results_capped: pd.DataFrame) -> Dict[str, str]:
        """
        Generate all available visualizations and return file paths.
        
        Args:
            results_uncapped: DataFrame with uncapped simulation results
            results_capped: DataFrame with capped simulation results
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        logger.info("🎨 Generating comprehensive visualization suite...")
        
        generated_files = {}
        
        try:
            # Generate individual visualizations
            generated_files['wage_comparison'] = self.compare_average_wages(
                results_uncapped, results_capped)
            
            generated_files['final_wage_comparison'] = self.plot_final_wage_comparison(
                results_uncapped, results_capped)
            
            generated_files['workforce_composition'] = self.plot_workforce_composition(
                results_uncapped, results_capped)
            
            generated_files['conversion_statistics'] = self.plot_conversion_statistics(
                results_capped)
            
            generated_files['dashboard'] = self.create_summary_dashboard(
                results_uncapped, results_capped)
            
            # Filter out empty paths
            generated_files = {k: v for k, v in generated_files.items() if v}
            
            logger.info(f"✅ Successfully generated {len(generated_files)} visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            raise
        
        return generated_files
    
    def generate_backlog_visualizations(self, backlog_uncapped: pd.DataFrame, 
                                      backlog_capped: pd.DataFrame) -> Dict[str, str]:
        """
        Generate backlog-specific visualizations (NEW FOR SPEC-7).
        
        Args:
            backlog_uncapped: DataFrame with uncapped backlog data
            backlog_capped: DataFrame with capped backlog data
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        logger.info("📊 Generating backlog comparison visualizations...")
        
        generated_files = {}
        
        try:
            # Generate backlog visualizations
            generated_files['backlog_comparison'] = self.compare_backlog_sizes(
                backlog_uncapped, backlog_capped)
            
            generated_files['backlog_interactive'] = self.backlog_bar_interactive(
                backlog_uncapped, backlog_capped)
            
            # Filter out empty paths
            generated_files = {k: v for k, v in generated_files.items() if v}
            
            logger.info(f"✅ Successfully generated {len(generated_files)} backlog visualizations")
            
        except Exception as e:
            logger.error(f"Error generating backlog visualizations: {e}")
            raise
        
        return generated_files

def format_currency(value: float) -> str:
    """Format currency values for display."""
    return f"${math.ceil(value):,.0f}"

def format_number(value: int) -> str:
    """Format large numbers with commas."""
    return f"{value:,}"

def validate_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """
    Validate that DataFrames have required columns for visualization.
    
    Args:
        df1: First DataFrame to validate
        df2: Second DataFrame to validate
        
    Returns:
        True if both DataFrames are valid
    """
    required_columns = ['year', 'total_workers', 'permanent_workers', 
                       'temporary_workers', 'avg_wage_total']
    
    for df in [df1, df2]:
        if df.empty:
            logger.error("DataFrame is empty")
            return False
            
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"DataFrame missing required columns: {missing_columns}")
            return False
    
    return True

def validate_backlog_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """
    Validate that backlog DataFrames have required columns for visualization (NEW FOR SPEC-7).
    
    Args:
        df1: First backlog DataFrame to validate
        df2: Second backlog DataFrame to validate
        
    Returns:
        True if both DataFrames are valid
    """
    required_columns = ['nationality', 'backlog_size', 'scenario']
    
    for df in [df1, df2]:
        if df.empty:
            logger.error("Backlog DataFrame is empty")
            return False
            
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Backlog DataFrame missing required columns: {missing_columns}")
            return False
    
    return True
