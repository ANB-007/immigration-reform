# tests/test_visualization.py
"""
Unit tests for the visualization module.
Tests chart generation, data validation, and file outputs.
NEW FOR SPEC-6: Enhanced visualization and comparative salary analysis.
"""

import pytest
import tempfile
import os
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Test if visualization modules are available
try:
    from src.simulation.visualization import (
        SimulationVisualizer, validate_dataframes, 
        format_currency, format_number
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

@pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="Visualization modules not available")
class TestVisualization:
    """Test cases for visualization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = SimulationVisualizer(output_dir=self.temp_dir, save_plots=True)
        
        # Create sample data
        years = list(range(2025, 2035))
        self.sample_uncapped = pd.DataFrame({
            'year': years,
            'total_workers': [1000 + i * 25 for i in range(10)],
            'permanent_workers': [950 + i * 22 for i in range(10)],
            'temporary_workers': [50 + i * 3 for i in range(10)],
            'avg_wage_total': [95000 + i * 2000 for i in range(10)],
            'avg_wage_permanent': [96000 + i * 2100 for i in range(10)],
            'avg_wage_temporary': [93000 + i * 1800 for i in range(10)],
            'converted_temps': [5 + i for i in range(10)]
        })
        
        self.sample_capped = pd.DataFrame({
            'year': years,
            'total_workers': [1000 + i * 23 for i in range(10)],
            'permanent_workers': [950 + i * 20 for i in range(10)],
            'temporary_workers': [50 + i * 3 for i in range(10)],
            'avg_wage_total': [95000 + i * 1800 for i in range(10)],
            'avg_wage_permanent': [96000 + i * 1900 for i in range(10)],
            'avg_wage_temporary': [93000 + i * 1600 for i in range(10)],
            'converted_temps': [3 + i for i in range(10)]
        })
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_visualizer_initialization(self):
        """Test SimulationVisualizer initialization."""
        assert self.visualizer.output_dir == Path(self.temp_dir)
        assert self.visualizer.save_plots is True
        assert self.visualizer.output_dir.exists()
    
    def test_validate_dataframes_valid(self):
        """Test dataframe validation with valid data."""
        assert validate_dataframes(self.sample_uncapped, self.sample_capped) is True
    
    def test_validate_dataframes_empty(self):
        """Test dataframe validation with empty data."""
        empty_df = pd.DataFrame()
        assert validate_dataframes(empty_df, self.sample_capped) is False
        assert validate_dataframes(self.sample_uncapped, empty_df) is False
    
    def test_validate_dataframes_missing_columns(self):
        """Test dataframe validation with missing columns."""
        incomplete_df = pd.DataFrame({
            'year': [2025, 2026],
            'total_workers': [1000, 1025]
            # Missing required columns
        })
        assert validate_dataframes(incomplete_df, self.sample_capped) is False
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_compare_average_wages(self, mock_savefig, mock_show):
        """Test wage comparison chart generation."""
        filename = self.visualizer.compare_average_wages(
            self.sample_uncapped, self.sample_capped
        )
        
        # Check that plot functions were called
        mock_show.assert_called_once()
        mock_savefig.assert_called_once()
        
        # Check that file path is returned
        assert filename.endswith('wage_comparison_over_time.png')
        assert self.temp_dir in filename
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_workforce_composition(self, mock_savefig, mock_show):
        """Test workforce composition chart generation."""
        filename = self.visualizer.plot_workforce_composition(
            self.sample_uncapped, self.sample_capped
        )
        
        mock_show.assert_called_once()
        mock_savefig.assert_called_once()
        assert filename.endswith('workforce_composition_comparison.png')
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_conversion_statistics(self, mock_savefig, mock_show):
        """Test conversion statistics chart generation."""
        filename = self.visualizer.plot_conversion_statistics(self.sample_capped)
        
        mock_show.assert_called_once()
        mock_savefig.assert_called_once()
        assert filename.endswith('conversion_statistics.png')
    
    def test_plot_conversion_statistics_no_data(self):
        """Test conversion statistics with missing data."""
        df_no_conversions = self.sample_capped.drop('converted_temps', axis=1)
        filename = self.visualizer.plot_conversion_statistics(df_no_conversions)
        
        # Should return empty string when no conversion data
        assert filename == ""
    
    @patch('matplotlib.pyplot.show')
    def test_final_wage_comparison_matplotlib_fallback(self, mock_show):
        """Test final wage comparison with matplotlib fallback."""
        # Mock plotly as unavailable
        with patch.dict('sys.modules', {'plotly.express': None}):
            with patch('src.simulation.visualization.PLOTLY_AVAILABLE', False):
                with patch('matplotlib.pyplot.savefig') as mock_savefig:
                    filename = self.visualizer.plot_final_wage_comparison(
                        self.sample_uncapped, self.sample_capped
                    )
                    
                    mock_show.assert_called_once()
                    mock_savefig.assert_called_once()
                    assert filename.endswith('final_wage_comparison.png')
    
    def test_generate_all_visualizations(self):
        """Test comprehensive visualization generation."""
        with patch.object(self.visualizer, 'compare_average_wages', return_value='wage_comp.png'):
            with patch.object(self.visualizer, 'plot_final_wage_comparison', return_value='final_comp.html'):
                with patch.object(self.visualizer, 'plot_workforce_composition', return_value='workforce.png'):
                    with patch.object(self.visualizer, 'plot_conversion_statistics', return_value='conversions.png'):
                        with patch.object(self.visualizer, 'create_summary_dashboard', return_value='dashboard.html'):
                            
                            generated_files = self.visualizer.generate_all_visualizations(
                                self.sample_uncapped, self.sample_capped
                            )
                            
                            # Check that all visualizations were generated
                            assert len(generated_files) == 5
                            assert 'wage_comparison' in generated_files
                            assert 'final_wage_comparison' in generated_files
                            assert 'workforce_composition' in generated_files
                            assert 'conversion_statistics' in generated_files
                            assert 'dashboard' in generated_files
    
    def test_visualizer_with_save_plots_disabled(self):
        """Test visualizer with save_plots disabled."""
        visualizer_no_save = SimulationVisualizer(
            output_dir=self.temp_dir, save_plots=False
        )
        
        with patch('matplotlib.pyplot.show'):
            filename = visualizer_no_save.compare_average_wages(
                self.sample_uncapped, self.sample_capped
            )
            
            # Should return empty string when save_plots is False
            assert filename == ""

class TestVisualizationUtilities:
    """Test utility functions in visualization module."""
    
    def test_format_currency(self):
        """Test currency formatting."""
        if not VISUALIZATION_AVAILABLE:
            pytest.skip("Visualization modules not available")
        
        assert format_currency(95000) == "$95,000"
        assert format_currency(95000.50) == "$95,001"  # Rounds to nearest dollar
        assert format_currency(1234567.89) == "$1,234,568"
    
    def test_format_number(self):
        """Test number formatting."""
        if not VISUALIZATION_AVAILABLE:
            pytest.skip("Visualization modules not available")
        
        assert format_number(1000) == "1,000"
        assert format_number(1234567) == "1,234,567"
        assert format_number(100) == "100"

@pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="Visualization modules not available")
class TestVisualizationIntegration:
    """Integration tests for visualization with actual data."""
    
    def test_real_data_visualization(self):
        """Test visualization with realistic simulation data."""
        # Create more realistic test data
        years = list(range(2025, 2031))  # 6 years
        realistic_uncapped = pd.DataFrame({
            'year': years,
            'total_workers': [10000, 10250, 10506, 10769, 11040, 11318],
            'permanent_workers': [9959, 10200, 10447, 10700, 10960, 11227],
            'temporary_workers': [41, 50, 59, 69, 80, 91],
            'avg_wage_total': [95000, 97080, 99244, 101495, 103837, 106275],
            'avg_wage_permanent': [95100, 97200, 99384, 101652, 104009, 106460],
            'avg_wage_temporary': [92000, 93840, 95722, 97638, 99591, 101583],
            'converted_temps': [8, 9, 10, 11, 12, 13]
        })
        
        realistic_capped = pd.DataFrame({
            'year': years,
            'total_workers': [10000, 10248, 10501, 10760, 11025, 11297],
            'permanent_workers': [9959, 10195, 10436, 10683, 10937, 11197],
            'temporary_workers': [41, 53, 65, 77, 88, 100],
            'avg_wage_total': [95000, 96950, 98950, 101002, 103107, 105267],
            'avg_wage_permanent': [95100, 97070, 99092, 101168, 103299, 105487],
            'avg_wage_temporary': [92000, 93680, 95402, 97167, 98976, 100830],
            'converted_temps': [5, 6, 6, 7, 7, 8]  # Lower due to per-country caps
        })
        
        # Validate that this data works with our validation
        assert validate_dataframes(realistic_uncapped, realistic_capped)
        
        # Test that visualization can handle this data without errors
        temp_dir = tempfile.mkdtemp()
        try:
            visualizer = SimulationVisualizer(output_dir=temp_dir, save_plots=False)
            
            with patch('matplotlib.pyplot.show'):
                # These should not raise exceptions
                visualizer.compare_average_wages(realistic_uncapped, realistic_capped)
                visualizer.plot_workforce_composition(realistic_uncapped, realistic_capped)
                visualizer.plot_conversion_statistics(realistic_capped)
                
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

def test_visualization_import():
    """Test that visualization modules can be imported correctly."""
    try:
        from src.simulation.visualization import SimulationVisualizer
        assert True  # Import successful
    except ImportError:
        # Check if required packages are available
        try:
            import matplotlib
            import seaborn
            import pandas
            # If we get here, the packages exist but there's another issue
            assert False, "Visualization packages available but module import failed"
        except ImportError:
            # Expected if packages aren't installed
            pytest.skip("Visualization packages not installed")

if __name__ == "__main__":
    pytest.main([__file__])
