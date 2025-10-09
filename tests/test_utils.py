# tests/test_utils.py
"""
Unit tests for utility functions.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.simulation.utils import (
    save_simulation_results, load_simulation_results, validate_configuration,
    format_number, format_percentage, calculate_compound_growth
)
from src.simulation.models import SimulationState, SimulationConfig

class TestUtils:
    """Test cases for utility functions."""

    def test_save_and_load_results(self):
        """Test saving and loading simulation results."""
        # Create test data
        states = [
            SimulationState(2025, 1000, 900, 100, 0, 0),
            SimulationState(2026, 1050, 940, 110, 40, 10),
            SimulationState(2027, 1100, 980, 120, 40, 10)
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            # Save results
            save_simulation_results(states, temp_path)

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

        finally:
            os.unlink(temp_path)

    def test_validate_configuration(self):
        """Test configuration validation."""
        # Valid config
        config = SimulationConfig(initial_workers=1000, years=10)
        errors = validate_configuration(config)
        assert len(errors) == 0

        # Invalid configs
        invalid_config = SimulationConfig(initial_workers=0, years=10)
        errors = validate_configuration(invalid_config)
        assert len(errors) > 0
        assert any("positive" in error for error in errors)

        # Test excessive years
        long_config = SimulationConfig(initial_workers=1000, years=100)
        errors = validate_configuration(long_config)
        assert len(errors) > 0
        assert any("exceed" in error for error in errors)

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

def test_file_operations():
    """Test file I/O operations work correctly."""
    # Test directory creation for output
    with tempfile.TemporaryDirectory() as temp_dir:
        nested_path = os.path.join(temp_dir, "nested", "dir", "output.csv")

        states = [SimulationState(2025, 100, 90, 10, 0, 0)]
        save_simulation_results(states, nested_path)

        assert os.path.exists(nested_path)

        loaded_states = load_simulation_results(nested_path)
        assert len(loaded_states) == 1
        assert loaded_states[0].year == 2025

if __name__ == "__main__":
    pytest.main([__file__])
