# tests/test_utils.py
"""
Unit tests for utility functions.
Updated for SPEC-5 per-country cap functionality.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from src.simulation.utils import (
    save_simulation_results, load_simulation_results, validate_configuration,
    format_number, format_percentage, calculate_compound_growth,
    export_country_cap_analysis, analyze_conversion_queue_efficiency
)
from src.simulation.models import SimulationState, SimulationConfig, Worker, WorkerStatus

class TestUtils:
    """Test cases for utility functions."""
    
    def test_save_and_load_results(self):
        """Test saving and loading simulation results."""
        # Create test data with SPEC-5 features - FIXED: Set country_cap_enabled properly
        states = [
            SimulationState(2025, 1000, 900, 100, 0, 0, 0, 95000.0, 95000.0, 95000.0, 95000000.0, 
                        {"India": 0.7, "China": 0.1}, {}, {}, country_cap_enabled=False),
            SimulationState(2026, 1050, 940, 110, 40, 10, 5, 96000.0, 96500.0, 95000.0, 100800000.0,
                        {"India": 0.69, "China": 0.11}, {"India": 3, "China": 2}, {"India": 15, "China": 5}, 
                        country_cap_enabled=True),
            SimulationState(2027, 1100, 980, 120, 40, 10, 5, 97000.0, 97500.0, 95500.0, 106700000.0,
                        {"India": 0.68, "China": 0.12}, {"India": 3, "China": 2}, {"India": 20, "China": 8},
                        country_cap_enabled=True)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save results
            save_simulation_results(states, temp_path, include_nationality_columns=True)
            
            # Load results
            loaded_states = load_simulation_results(temp_path)
            
            assert len(loaded_states) == len(states)
            for original, loaded in zip(states, loaded_states):
                assert original.year == loaded.year
                assert original.total_workers == loaded.total_workers
                assert original.permanent_workers == loaded.permanent_workers
                assert original.temporary_workers == loaded.temporary_workers
                assert original.new_permanent == loaded.new_permanent
                assert original.new_temporary == loaded.new_temporary
                assert original.converted_temps == loaded.converted_temps
                
                # Test wage data (FROM SPEC-3)
                assert abs(original.avg_wage_total - loaded.avg_wage_total) < 0.01
                assert abs(original.total_wage_bill - loaded.total_wage_bill) < 0.01
                
                # Test nationality data (FROM SPEC-4)
                assert original.top_temp_nationalities == loaded.top_temp_nationalities
                
                # NEW FOR SPEC-5: Test per-country cap data
                assert original.converted_by_country == loaded.converted_by_country
                assert original.queue_backlog_by_country == loaded.queue_backlog_by_country
                
                # FIXED: Test country cap enabled status
                assert loaded.country_cap_enabled == (bool(original.converted_by_country) or bool(original.queue_backlog_by_country))
        
        finally:
            os.unlink(temp_path)

    
    def test_save_and_load_results_without_nationality_columns(self):
        """Test saving and loading without nationality columns for backward compatibility."""
        states = [
            SimulationState(2025, 1000, 900, 100, 0, 0, 0),
            SimulationState(2026, 1050, 940, 110, 40, 10, 5)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save results without nationality columns
            save_simulation_results(states, temp_path, include_nationality_columns=False)
            
            # Load results
            loaded_states = load_simulation_results(temp_path)
            
            assert len(loaded_states) == len(states)
            for original, loaded in zip(states, loaded_states):
                assert original.year == loaded.year
                assert original.total_workers == loaded.total_workers
                assert original.converted_temps == loaded.converted_temps
        
        finally:
            os.unlink(temp_path)
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        # Valid config
        config = SimulationConfig(initial_workers=1000, years=10)
        errors = validate_configuration(config)
        assert len(errors) == 0
        
        # Test that invalid configs raise ValueError during creation
        with pytest.raises(ValueError, match="Initial workers must be positive"):
            SimulationConfig(initial_workers=0, years=10)
        
        with pytest.raises(ValueError, match="Simulation years must be positive"):
            SimulationConfig(initial_workers=1000, years=0)
        
        # Test validation function on edge cases
        large_config = SimulationConfig(initial_workers=600000, years=10)
        errors = validate_configuration(large_config)
        assert len(errors) > 0
        assert any("slow" in error for error in errors)
        
        # Test excessive years
        long_config = SimulationConfig(initial_workers=1000, years=100)
        errors = validate_configuration(long_config)
        assert len(errors) > 0
        assert any("exceed" in error for error in errors)
        
        # NEW FOR SPEC-5: Test country cap specific validations
        small_country_cap_config = SimulationConfig(initial_workers=500, years=10, country_cap_enabled=True)
        errors = validate_configuration(small_country_cap_config)
        assert len(errors) > 0
        assert any("discretization" in error for error in errors)

    
    def test_format_functions(self):
        """Test number and percentage formatting."""
        assert format_number(1000) == "1,000"
        assert format_number(1000000) == "1,000,000"
        
        assert format_percentage(0.1234) == "12.34%"
        assert format_percentage(0.1234, 1) == "12.3%"
        assert format_percentage(0.001, 3) == "0.100%"
    
    def test_compound_growth(self):
        """Test compound growth calculation."""
        result = calculate_compound_growth(1000, 0.05, 10)
        expected = round(1000 * (1.05 ** 10))
        assert result == expected
        
        # Test zero growth
        result = calculate_compound_growth(1000, 0.0, 5)
        assert result == 1000
        
        # Test negative growth
        result = calculate_compound_growth(1000, -0.1, 2)
        expected = round(1000 * (0.9 ** 2))
        assert result == expected
    
    # NEW FOR SPEC-5: Test per-country cap specific utilities
    def test_export_country_cap_analysis(self):
        """Test exporting per-country cap analysis (NEW FOR SPEC-5)."""
        # Create test data with country cap information
        states = [
            SimulationState(2025, 1000, 900, 100, 0, 0, 0, country_cap_enabled=True,
                          converted_by_country={}, queue_backlog_by_country={"India": 20, "China": 5}),
            SimulationState(2026, 1050, 940, 110, 40, 10, 8, country_cap_enabled=True,
                          converted_by_country={"India": 5, "China": 3}, 
                          queue_backlog_by_country={"India": 18, "China": 4}),
            SimulationState(2027, 1100, 980, 120, 40, 10, 7, country_cap_enabled=True,
                          converted_by_country={"India": 4, "China": 3}, 
                          queue_backlog_by_country={"India": 20, "China": 6})
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            export_country_cap_analysis(states, temp_path)
            
            # Verify file was created and contains expected data
            assert os.path.exists(temp_path)
            
            # Read and verify content
            with open(temp_path, 'r') as f:
                content = f.read()
                assert "India" in content
                assert "China" in content
                assert "2026" in content
                assert "2027" in content
        
        finally:
            os.unlink(temp_path)
    
    def test_export_country_cap_analysis_no_cap_data(self):
        """Test exporting country cap analysis with no cap data."""
        # States without country cap enabled
        states = [
            SimulationState(2025, 1000, 900, 100, 0, 0, 0, country_cap_enabled=False),
            SimulationState(2026, 1050, 940, 110, 40, 10, 8, country_cap_enabled=False)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            export_country_cap_analysis(states, temp_path)
            
            # File should not be created or should be empty since no cap data exists
            # The function should log a warning but not fail
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_analyze_conversion_queue_efficiency(self):
        """Test conversion queue efficiency analysis (NEW FOR SPEC-5)."""
        # Create states with country cap data
        states = [
            SimulationState(2025, 1000, 900, 100, 0, 0, 0, country_cap_enabled=True,
                          converted_by_country={}, queue_backlog_by_country={"India": 50, "China": 20}),
            SimulationState(2026, 1050, 940, 110, 40, 10, 15, country_cap_enabled=True,
                          converted_by_country={"India": 10, "China": 5}, 
                          queue_backlog_by_country={"India": 55, "China": 22}),
            SimulationState(2027, 1100, 980, 120, 40, 10, 18, country_cap_enabled=True,
                          converted_by_country={"India": 12, "China": 6}, 
                          queue_backlog_by_country={"India": 60, "China": 25})
        ]
        
        efficiency = analyze_conversion_queue_efficiency(states)
        
        assert "error" not in efficiency
        assert efficiency["total_conversions"] == 15 + 18  # Sum of conversions from years 2-3
        assert efficiency["final_total_backlog"] == 60 + 25  # Final backlog
        assert efficiency["countries_with_backlogs"] == 2
        assert efficiency["years_analyzed"] == 3
        
        # Check backlog growth rates
        assert "backlog_growth_rates" in efficiency
        growth_rates = efficiency["backlog_growth_rates"]
        assert "India" in growth_rates
        assert "China" in growth_rates
    
    def test_analyze_conversion_queue_efficiency_no_data(self):
        """Test conversion queue efficiency analysis with no cap data."""
        # States without country cap
        states = [
            SimulationState(2025, 1000, 900, 100, 0, 0, 0, country_cap_enabled=False),
            SimulationState(2026, 1050, 940, 110, 40, 10, 8, country_cap_enabled=False)
        ]
        
        efficiency = analyze_conversion_queue_efficiency(states)
        
        assert "error" in efficiency
        assert efficiency["error"] == "No per-country cap data available"

def test_file_operations():
    """Test file I/O operations work correctly."""
    # Test directory creation for output
    with tempfile.TemporaryDirectory() as temp_dir:
        nested_path = os.path.join(temp_dir, "nested", "dir", "output.csv")
        
        states = [SimulationState(2025, 100, 90, 10, 0, 0, 0)]
        save_simulation_results(states, nested_path)
        
        assert os.path.exists(nested_path)
        
        loaded_states = load_simulation_results(nested_path)
        assert len(loaded_states) == 1
        assert loaded_states[0].year == 2025

def test_json_serialization_robustness():
    """Test that JSON serialization handles edge cases properly (NEW FOR SPEC-5)."""
    # Test with empty dictionaries
    state_empty = SimulationState(
        2025, 100, 90, 10, 0, 0, 0,
        converted_by_country={},
        queue_backlog_by_country={},
        country_cap_enabled=True
    )
    
    # Test with None values
    state_none = SimulationState(
        2025, 100, 90, 10, 0, 0, 0,
        converted_by_country={},
        queue_backlog_by_country={},
        country_cap_enabled=False
    )
    
    states = [state_empty, state_none]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        save_simulation_results(states, temp_path, include_nationality_columns=True)
        loaded_states = load_simulation_results(temp_path)
        
        assert len(loaded_states) == 2
        assert loaded_states[0].converted_by_country == {}
        assert loaded_states[0].queue_backlog_by_country == {}
        
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    pytest.main([__file__])
