# tests/test_simulation.py
"""
Unit tests for the core simulation functionality.
"""

import pytest
import numpy as np
from src.simulation.models import SimulationConfig, SimulationState, Worker, WorkerStatus
from src.simulation.sim import Simulation
from src.simulation.empirical_params import H1B_SHARE, ANNUAL_PERMANENT_ENTRY_RATE

class TestSimulation:
    """Test cases for the Simulation class."""

    def test_initialization(self):
        """Test simulation initialization."""
        config = SimulationConfig(initial_workers=1000, years=5, seed=42)
        sim = Simulation(config)

        assert len(sim.states) == 1
        assert sim.states[0].total_workers == 1000
        assert sim.states[0].year == 2025

        # Check H-1B share is approximately correct
        expected_h1b = round(1000 * H1B_SHARE)
        assert sim.states[0].temporary_workers == expected_h1b
        assert sim.states[0].permanent_workers == 1000 - expected_h1b

    def test_single_step(self):
        """Test a single simulation step."""
        config = SimulationConfig(initial_workers=1000, years=1, seed=42)
        sim = Simulation(config)

        initial_total = sim.states[0].total_workers
        next_state = sim.step()

        assert next_state.year == 2026
        assert next_state.total_workers > initial_total
        assert next_state.new_permanent > 0
        assert next_state.new_temporary >= 0

    def test_full_run(self):
        """Test a full simulation run."""
        config = SimulationConfig(initial_workers=10000, years=10, seed=42)
        sim = Simulation(config)

        states = sim.run()

        assert len(states) == 11  # Initial + 10 years
        assert states[-1].year == 2035
        assert states[-1].total_workers > states[0].total_workers

    def test_proportional_growth(self):
        """Test that doubling initial population doubles final output."""
        config1 = SimulationConfig(initial_workers=5000, years=5, seed=42)
        config2 = SimulationConfig(initial_workers=10000, years=5, seed=42)

        sim1 = Simulation(config1)
        sim2 = Simulation(config2)

        states1 = sim1.run()
        states2 = sim2.run()

        # Final totals should be approximately proportional
        ratio = states2[-1].total_workers / states1[-1].total_workers
        assert 1.9 < ratio < 2.1  # Allow for rounding differences

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        config1 = SimulationConfig(initial_workers=1000, years=5, seed=123)
        config2 = SimulationConfig(initial_workers=1000, years=5, seed=123)

        sim1 = Simulation(config1)
        sim2 = Simulation(config2)

        states1 = sim1.run()
        states2 = sim2.run()

        for i in range(len(states1)):
            assert states1[i].total_workers == states2[i].total_workers
            assert states1[i].permanent_workers == states2[i].permanent_workers

    def test_small_population_warning(self, caplog):
        """Test warning for small initial populations."""
        config = SimulationConfig(initial_workers=50, years=1, seed=42)
        sim = Simulation(config)

        assert "Small initial population" in caplog.text

    def test_agent_model_conversion(self):
        """Test conversion to agent-based model."""
        config = SimulationConfig(initial_workers=100, years=1, seed=42)
        sim = Simulation(config)
        sim.run()

        workers = sim.to_agent_model()

        assert len(workers) == sim.states[-1].total_workers

        permanent_count = sum(1 for w in workers if w.is_permanent)
        temporary_count = sum(1 for w in workers if w.is_temporary)

        assert permanent_count == sim.states[-1].permanent_workers
        assert temporary_count == sim.states[-1].temporary_workers

class TestModels:
    """Test cases for data models."""

    def test_worker_creation(self):
        """Test Worker model creation and validation."""
        worker = Worker(id=1, status=WorkerStatus.PERMANENT, age=30)

        assert worker.is_permanent
        assert not worker.is_temporary
        assert worker.age == 30

    def test_worker_invalid_age(self):
        """Test Worker age validation."""
        with pytest.raises(ValueError):
            Worker(id=1, status=WorkerStatus.PERMANENT, age=150)

    def test_simulation_state_validation(self):
        """Test SimulationState validation."""
        # Valid state
        state = SimulationState(
            year=2025, total_workers=100, 
            permanent_workers=90, temporary_workers=10
        )
        assert state.h1b_share == 0.1

        # Invalid state - workers don't sum correctly
        with pytest.raises(ValueError):
            SimulationState(
                year=2025, total_workers=100,
                permanent_workers=80, temporary_workers=30
            )

    def test_configuration_validation(self):
        """Test SimulationConfig validation."""
        # Valid config
        config = SimulationConfig(initial_workers=1000, years=10)
        assert config.initial_workers == 1000

        # Invalid configs
        with pytest.raises(ValueError):
            SimulationConfig(initial_workers=0, years=10)

        with pytest.raises(ValueError):
            SimulationConfig(initial_workers=1000, years=0)

def test_growth_rates():
    """Test that growth rates are reasonable."""
    config = SimulationConfig(initial_workers=100000, years=1, seed=42)
    sim = Simulation(config)
    sim.run()

    growth_rate = sim.get_growth_rate(1)

    # Should be roughly equal to sum of entry rates
    expected_rate = ANNUAL_PERMANENT_ENTRY_RATE + (H1B_SHARE * ANNUAL_PERMANENT_ENTRY_RATE) / (1 - H1B_SHARE)

    # Allow for rounding differences
    assert abs(growth_rate - expected_rate) < 0.005

if __name__ == "__main__":
    pytest.main([__file__])
