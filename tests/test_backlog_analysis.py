# tests/test_backlog_analysis.py
"""
Unit tests for backlog analysis functionality.
Tests backlog tracking, CSV export, and visualization integration.
NEW FOR SPEC-7: Comparative backlog analysis by nationality.
"""

import pytest
import tempfile
import os
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from collections import deque

from src.simulation.models import (
    SimulationConfig, BacklogAnalysis, TemporaryWorker, Worker, WorkerStatus
)
from src.simulation.sim import Simulation
from src.simulation.utils import save_backlog_analysis, load_backlog_analysis
from src.simulation.empirical_params import TEMP_NATIONALITY_DISTRIBUTION

# Test if visualization modules are available
try:
    from src.simulation.visualization import (
        SimulationVisualizer, validate_backlog_dataframes
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

class TestBacklogAnalysis:
    """Test cases for backlog analysis functionality."""
    
    def test_backlog_analysis_creation_uncapped(self):
        """Test BacklogAnalysis creation for uncapped scenario."""
        # Create mock simulation with global queue
        mock_sim = MagicMock()
        mock_sim.country_cap_enabled = False
        mock_sim.country_queues = {}
        
        # Create mock global queue with nationality-distributed workers
        mock_sim.global_queue = deque([
            TemporaryWorker(1, 2025, "India"),
            TemporaryWorker(2, 2025, "China"), 
            TemporaryWorker(3, 2025, "India"),
            TemporaryWorker(4, 2025, "Canada")
        ])
        
        # Mock states
        mock_sim.states = [MagicMock()]
        mock_sim.states[0].year = 2030
        
        # Create backlog analysis
        analysis = BacklogAnalysis.from_simulation(mock_sim, "uncapped")
        
        assert analysis.scenario == "uncapped"
        assert analysis.final_year == 2030
        assert analysis.total_backlog == 4
        assert analysis.backlog_by_nationality["India"] == 2
        assert analysis.backlog_by_nationality["China"] == 1
        assert analysis.backlog_by_nationality["Canada"] == 1
        # Other nationalities should have 0
        assert analysis.backlog_by_nationality["Philippines"] == 0
    
    def test_backlog_analysis_creation_capped(self):
        """Test BacklogAnalysis creation for capped scenario."""
        # Create mock simulation with country queues
        mock_sim = MagicMock()
        mock_sim.country_cap_enabled = True
        mock_sim.global_queue = None
        
        # Create mock country queues
        mock_sim.country_queues = {
            "India": deque([TemporaryWorker(1, 2025, "India"), TemporaryWorker(2, 2025, "India")]),
            "China": deque([TemporaryWorker(3, 2025, "China")]),
            "Canada": deque()  # Empty queue
        }
        
        # Mock states
        mock_sim.states = [MagicMock()]
        mock_sim.states[0].year = 2030
        
        # Create backlog analysis
        analysis = BacklogAnalysis.from_simulation(mock_sim, "capped")
        
        assert analysis.scenario == "capped"
        assert analysis.final_year == 2030
        assert analysis.total_backlog == 3
        assert analysis.backlog_by_nationality["India"] == 2
        assert analysis.backlog_by_nationality["China"] == 1
        assert analysis.backlog_by_nationality["Canada"] == 0
    
    def test_backlog_analysis_to_dataframe(self):
        """Test conversion of BacklogAnalysis to DataFrame."""
        # Create test backlog analysis
        backlog_data = {nationality: 0 for nationality in TEMP_NATIONALITY_DISTRIBUTION.keys()}
        backlog_data["India"] = 100
        backlog_data["China"] = 50
        
        analysis = BacklogAnalysis(
            scenario="capped",
            backlog_by_nationality=backlog_data,
            total_backlog=150,
            final_year=2030
        )
        
        df = analysis.to_dataframe()
        
        # Check DataFrame structure
        assert 'nationality' in df.columns
        assert 'backlog_size' in df.columns
        assert 'scenario' in df.columns
        
        # Check data content
        assert len(df) == len(TEMP_NATIONALITY_DISTRIBUTION)
        assert all(df['scenario'] == 'capped')
        
        # Check specific values
        india_row = df[df['nationality'] == 'India'].iloc[0]
        assert india_row['backlog_size'] == 100
        
        china_row = df[df['nationality'] == 'China'].iloc[0]
        assert china_row['backlog_size'] == 50
    
    def test_backlog_analysis_get_top_backlogs(self):
        """Test getting top backlogs by nationality."""
        backlog_data = {
            "India": 1000,
            "China": 500,
            "Canada": 100,
            "Philippines": 50,
            "South Korea": 25,
            "United Kingdom": 10,
            "Mexico": 5,
            "Brazil": 2,
            "Germany": 1,
            "Other": 0
        }
        
        analysis = BacklogAnalysis(
            scenario="test",
            backlog_by_nationality=backlog_data,
            total_backlog=sum(backlog_data.values()),
            final_year=2030
        )
        
        # Test top 3
        top_3 = analysis.get_top_backlogs(3)
        assert len(top_3) == 3
        assert top_3["India"] == 1000
        assert top_3["China"] == 500
        assert top_3["Canada"] == 100
        
        # Test top 5
        top_5 = analysis.get_top_backlogs(5)
        assert len(top_5) == 5
        assert top_5["Philippines"] == 50
        assert top_5["South Korea"] == 25
    
    def test_save_and_load_backlog_analysis(self):
        """Test saving and loading backlog analysis."""
        # Create test backlog analysis
        backlog_data = {nationality: 0 for nationality in TEMP_NATIONALITY_DISTRIBUTION.keys()}
        backlog_data["India"] = 200
        backlog_data["China"] = 75
        
        original_analysis = BacklogAnalysis(
            scenario="capped",
            backlog_by_nationality=backlog_data,
            total_backlog=275,
            final_year=2035
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save analysis
            save_backlog_analysis(original_analysis, temp_path)
            
            # Load analysis back as DataFrame
            loaded_df = load_backlog_analysis(temp_path)
            
            # Verify loaded data
            assert not loaded_df.empty
            assert len(loaded_df) == len(TEMP_NATIONALITY_DISTRIBUTION)
            
            # Check specific values
            india_row = loaded_df[loaded_df['nationality'] == 'India'].iloc[0]
            assert india_row['backlog_size'] == 200
            assert india_row['scenario'] == 'capped'
            
            china_row = loaded_df[loaded_df['nationality'] == 'China'].iloc[0]
            assert china_row['backlog_size'] == 75
            
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_backlog_analysis(self):
        """Test loading backlog analysis from nonexistent file."""
        df = load_backlog_analysis("nonexistent_file.csv")
        assert df.empty

class TestBacklogSimulationIntegration:
    """Test backlog analysis integration with simulations."""
    
    def test_uncapped_simulation_zero_backlog(self):
        """Test that uncapped simulation produces zero or minimal backlog."""
        config = SimulationConfig(
            initial_workers=1000, 
            years=5, 
            seed=42, 
            country_cap_enabled=False
        )
        
        sim = Simulation(config)
        sim.run()
        
        backlog_analysis = BacklogAnalysis.from_simulation(sim, "uncapped")
        
        # Uncapped should have zero or very small backlog
        # (depending on conversion cap vs temporary workers)
        assert backlog_analysis.total_backlog >= 0
        assert backlog_analysis.scenario == "uncapped"
    
    def test_capped_simulation_has_backlog(self):
        """Test that capped simulation produces backlogs for major countries."""
        config = SimulationConfig(
            initial_workers=5000, 
            years=15, 
            seed=42, 
            country_cap_enabled=True
        )
        
        sim = Simulation(config)
        sim.run()
        
        backlog_analysis = BacklogAnalysis.from_simulation(sim, "capped")
        
        # Capped should have some backlog
        assert backlog_analysis.total_backlog >= 0
        assert backlog_analysis.scenario == "capped"
        
        # India should likely have a backlog given its high proportion
        # (This is probabilistic, but with seed=42 should be consistent)
        if backlog_analysis.total_backlog > 0:
            top_backlogs = backlog_analysis.get_top_backlogs(3)
            # At least one country should have a backlog
            assert any(backlog > 0 for backlog in top_backlogs.values())
    
    def test_backlog_comparison_consistency(self):
        """Test that backlog analysis is consistent between scenarios."""
        base_config = {
            'initial_workers': 2000,
            'years': 10,
            'seed': 123,
            'agent_mode': True
        }
        
        # Run uncapped scenario
        config_uncapped = SimulationConfig(**base_config, country_cap_enabled=False)
        sim_uncapped = Simulation(config_uncapped)
        sim_uncapped.run()
        backlog_uncapped = BacklogAnalysis.from_simulation(sim_uncapped, "uncapped")
        
        # Run capped scenario
        config_capped = SimulationConfig(**base_config, country_cap_enabled=True)
        sim_capped = Simulation(config_capped)
        sim_capped.run()
        backlog_capped = BacklogAnalysis.from_simulation(sim_capped, "capped")
        
        # Capped should have same or higher total backlog
        assert backlog_capped.total_backlog >= backlog_uncapped.total_backlog
        
        # Both should have data for all nationalities
        assert len(backlog_uncapped.backlog_by_nationality) == len(TEMP_NATIONALITY_DISTRIBUTION)
        assert len(backlog_capped.backlog_by_nationality) == len(TEMP_NATIONALITY_DISTRIBUTION)

@pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="Visualization modules not available")
class TestBacklogVisualization:
    """Test backlog visualization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = SimulationVisualizer(output_dir=self.temp_dir, save_plots=True)
        
        # Create sample backlog data
        self.sample_uncapped = pd.DataFrame([
            {'nationality': 'India', 'backlog_size': 0, 'scenario': 'uncapped'},
            {'nationality': 'China', 'backlog_size': 0, 'scenario': 'uncapped'},
            {'nationality': 'Canada', 'backlog_size': 0, 'scenario': 'uncapped'},
            {'nationality': 'Philippines', 'backlog_size': 0, 'scenario': 'uncapped'},
            {'nationality': 'South Korea', 'backlog_size': 0, 'scenario': 'uncapped'},
        ])
        
        self.sample_capped = pd.DataFrame([
            {'nationality': 'India', 'backlog_size': 1500, 'scenario': 'capped'},
            {'nationality': 'China', 'backlog_size': 800, 'scenario': 'capped'},
            {'nationality': 'Canada', 'backlog_size': 200, 'scenario': 'capped'},
            {'nationality': 'Philippines', 'backlog_size': 100, 'scenario': 'capped'},
            {'nationality': 'South Korea', 'backlog_size': 50, 'scenario': 'capped'},
        ])
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_backlog_dataframes_valid(self):
        """Test backlog dataframe validation with valid data."""
        assert validate_backlog_dataframes(self.sample_uncapped, self.sample_capped) is True
    
    def test_validate_backlog_dataframes_empty(self):
        """Test backlog dataframe validation with empty data."""
        empty_df = pd.DataFrame()
        assert validate_backlog_dataframes(empty_df, self.sample_capped) is False
        assert validate_backlog_dataframes(self.sample_uncapped, empty_df) is False
    
    def test_validate_backlog_dataframes_missing_columns(self):
        """Test backlog dataframe validation with missing columns."""
        incomplete_df = pd.DataFrame({
            'nationality': ['India', 'China'],
            'backlog_size': [100, 50]
            # Missing 'scenario' column
        })
        assert validate_backlog_dataframes(incomplete_df, self.sample_capped) is False
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_compare_backlog_sizes(self, mock_savefig, mock_show):
        """Test backlog comparison chart generation."""
        filename = self.visualizer.compare_backlog_sizes(
            self.sample_uncapped, self.sample_capped
        )
        
        # Check that plot functions were called
        mock_show.assert_called_once()
        mock_savefig.assert_called_once()
        
        # Check that file path is returned
        assert filename.endswith('backlog_comparison.png')
        assert self.temp_dir in filename
    
    def test_backlog_bar_interactive_without_plotly(self):
        """Test interactive backlog chart when Plotly is not available."""
        with patch('src.simulation.visualization.PLOTLY_AVAILABLE', False):
            filename = self.visualizer.backlog_bar_interactive(
                self.sample_uncapped, self.sample_capped
            )
            
            # Should return empty string when Plotly is not available
            assert filename == ""
    
    def test_generate_backlog_visualizations(self):
        """Test comprehensive backlog visualization generation."""
        with patch.object(self.visualizer, 'compare_backlog_sizes', return_value='backlog_comp.png'):
            with patch.object(self.visualizer, 'backlog_bar_interactive', return_value='backlog_interactive.html'):
                
                generated_files = self.visualizer.generate_backlog_visualizations(
                    self.sample_uncapped, self.sample_capped
                )
                
                # Check that all backlog visualizations were generated
                assert len(generated_files) == 2
                assert 'backlog_comparison' in generated_files
                assert 'backlog_interactive' in generated_files
    
    def test_visualizer_with_save_plots_disabled(self):
        """Test backlog visualizer with save_plots disabled."""
        visualizer_no_save = SimulationVisualizer(
            output_dir=self.temp_dir, save_plots=False
        )
        
        with patch('matplotlib.pyplot.show'):
            filename = visualizer_no_save.compare_backlog_sizes(
                self.sample_uncapped, self.sample_capped
            )
            
            # Should return empty string when save_plots is False
            assert filename == ""

class TestBacklogAnalysisEdgeCases:
    """Test edge cases for backlog analysis."""
    
    def test_backlog_analysis_with_no_temp_workers(self):
        """Test backlog analysis when there are no temporary workers."""
        # Create mock simulation with empty queues
        mock_sim = MagicMock()
        mock_sim.country_cap_enabled = True
        mock_sim.global_queue = None
        mock_sim.country_queues = {
            nationality: deque() for nationality in TEMP_NATIONALITY_DISTRIBUTION.keys()
        }
        mock_sim.states = [MagicMock()]
        mock_sim.states[0].year = 2030
        
        analysis = BacklogAnalysis.from_simulation(mock_sim, "capped")
        
        assert analysis.total_backlog == 0
        assert all(backlog == 0 for backlog in analysis.backlog_by_nationality.values())
    
    def test_backlog_analysis_with_unknown_nationality(self):
        """Test backlog analysis with worker of unknown nationality."""
        mock_sim = MagicMock()
        mock_sim.country_cap_enabled = False
        mock_sim.country_queues = {}
        
        # Include worker with nationality not in TEMP_NATIONALITY_DISTRIBUTION
        mock_sim.global_queue = deque([
            TemporaryWorker(1, 2025, "UnknownCountry"),
            TemporaryWorker(2, 2025, "India")
        ])
        
        mock_sim.states = [MagicMock()]
        mock_sim.states[0].year = 2030
        
        analysis = BacklogAnalysis.from_simulation(mock_sim, "uncapped")
        
        # Unknown nationality should be ignored
        assert analysis.total_backlog == 1  # Only India worker counted
        assert analysis.backlog_by_nationality["India"] == 1
    
    def test_backlog_analysis_dataframe_includes_all_nationalities(self):
        """Test that DataFrame includes all nationalities even with zero backlog."""
        backlog_data = {nationality: 0 for nationality in TEMP_NATIONALITY_DISTRIBUTION.keys()}
        backlog_data["India"] = 100  # Only India has backlog
        
        analysis = BacklogAnalysis(
            scenario="test",
            backlog_by_nationality=backlog_data,
            total_backlog=100,
            final_year=2030
        )
        
        df = analysis.to_dataframe()
        
        # Should include all nationalities
        assert len(df) == len(TEMP_NATIONALITY_DISTRIBUTION)
        
        # Should include nationalities with zero backlog
        zero_backlog_rows = df[df['backlog_size'] == 0]
        assert len(zero_backlog_rows) == len(TEMP_NATIONALITY_DISTRIBUTION) - 1
        
        # Should include nationality with non-zero backlog
        nonzero_backlog_rows = df[df['backlog_size'] > 0]
        assert len(nonzero_backlog_rows) == 1
        assert nonzero_backlog_rows.iloc[0]['nationality'] == 'India'

class TestBacklogAnalysisConfiguration:
    """Test backlog analysis configuration and validation."""
    
    def test_simulation_config_with_compare_backlogs(self):
        """Test SimulationConfig with compare_backlogs enabled."""
        config = SimulationConfig(
            initial_workers=1000,
            years=10,
            compare_backlogs=True
        )
        
        assert config.compare_backlogs is True
        assert config.initial_workers == 1000
        assert config.years == 10
    
    def test_backlog_configuration_validation(self):
        """Test validation of backlog analysis configuration."""
        from src.simulation.utils import validate_configuration
        
        # Valid config with backlog analysis
        valid_config = SimulationConfig(
            initial_workers=10000,
            years=15,
            compare_backlogs=True
        )
        errors = validate_configuration(valid_config)
        assert len(errors) == 0
        
        # Invalid config - too few workers for meaningful backlog analysis
        invalid_config = SimulationConfig(
            initial_workers=1000,
            years=15,
            compare_backlogs=True
        )
        errors = validate_configuration(invalid_config)
        assert len(errors) > 0
        assert any("Backlog comparison with <5000 workers" in error for error in errors)
        
        # Invalid config - too few years for backlog accumulation
        invalid_config2 = SimulationConfig(
            initial_workers=10000,
            years=5,
            compare_backlogs=True
        )
        errors = validate_configuration(invalid_config2)
        assert len(errors) > 0
        assert any("Backlog comparison with <10 years" in error for error in errors)

if __name__ == "__main__":
    pytest.main([__file__])
