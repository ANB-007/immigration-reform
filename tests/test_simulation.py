# tests/test_simulation.py
"""
Unit tests for the core simulation functionality.
Updated for SPEC-2 temporary-to-permanent conversion testing.
"""

import pytest
import numpy as np
from src.simulation.models import SimulationConfig, SimulationState, Worker, WorkerStatus, TemporaryWorker
from src.simulation.sim import Simulation
from src.simulation.empirical_params import H1B_SHARE, ANNUAL_PERMANENT_ENTRY_RATE, GREEN_CARD_CAP_ABS, REAL_US_WORKFORCE_SIZE

class TestSimulation:
    """Test cases for the Simulation class."""
    
    def test_initialization(self):
        """Test simulation initialization."""
        config = SimulationConfig(initial_workers=1000, years=5, seed=42)
        sim = Simulation(config)
        
        assert len(sim.states) == 1
        assert sim.states[0].total_workers == 1000
        assert sim.states[0].year == 2025
        assert sim.states[0].converted_temps == 0  # NEW FOR SPEC-2
        
        # Check H-1B share is approximately correct
        expected_h1b = round(1000 * H1B_SHARE)
        assert sim.states[0].temporary_workers == expected_h1b
        assert sim.states[0].permanent_workers == 1000 - expected_h1b
        
        # NEW FOR SPEC-2: Check conversion cap calculation
        expected_cap = round(1000 * (GREEN_CARD_CAP_ABS / REAL_US_WORKFORCE_SIZE))
        assert sim.annual_conversion_cap == max(1, expected_cap)
    
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
        assert next_state.converted_temps >= 0  # NEW FOR SPEC-2
        
        # NEW FOR SPEC-2: Check conversion cap is respected
        assert next_state.converted_temps <= sim.annual_conversion_cap
    
    def test_full_run(self):
        """Test a full simulation run."""
        config = SimulationConfig(initial_workers=10000, years=10, seed=42)
        sim = Simulation(config)
        
        states = sim.run()
        
        assert len(states) == 11  # Initial + 10 years
        assert states[-1].year == 2035
        assert states[-1].total_workers > states[0].total_workers
        
        # NEW FOR SPEC-2: Check that conversions occurred
        total_conversions = sum(state.converted_temps for state in states[1:])
        assert total_conversions > 0
    
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
        
        # NEW FOR SPEC-2: Conversion caps should also be proportional
        assert sim2.annual_conversion_cap == 2 * sim1.annual_conversion_cap
    
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
            assert states1[i].converted_temps == states2[i].converted_temps  # NEW FOR SPEC-2
    
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

    # NEW FOR SPEC-2: Conversion-specific tests
    def test_conversion_cap_enforcement(self):
        """Test that conversion cap is never exceeded."""
        config = SimulationConfig(initial_workers=10000, years=20, seed=42)
        sim = Simulation(config)
        states = sim.run()
        
        for state in states[1:]:  # Skip initial state
            assert state.converted_temps <= sim.annual_conversion_cap
    
    def test_fifo_conversion_order(self):
        """Test that conversions follow FIFO order."""
        config = SimulationConfig(initial_workers=100, years=5, seed=42)
        sim = Simulation(config)
        
        # Check initial queue has workers from initial year
        initial_temp_workers = len(sim.temp_worker_queue)
        assert initial_temp_workers > 0
        
        # All initial temporary workers should have year_joined = 2025
        for temp_worker in sim.temp_worker_queue:
            assert temp_worker.year_joined == 2025
    
    def test_total_worker_consistency(self):
        """Test that total workers remain consistent after conversions."""
        config = SimulationConfig(initial_workers=5000, years=10, seed=42)
        sim = Simulation(config)
        states = sim.run()
        
        for i, state in enumerate(states[1:], 1):
            prev_state = states[i-1]
            expected_total = (prev_state.total_workers + 
                            state.new_permanent + state.new_temporary)
            assert state.total_workers == expected_total
    
    def test_conversion_utilization(self):
        """Test conversion utilization calculation."""
        config = SimulationConfig(initial_workers=10000, years=10, seed=42)
        sim = Simulation(config)
        sim.run()
        
        stats = sim.get_summary_stats()
        assert 0 <= stats['conversion_utilization'] <= 1
        assert stats['total_conversions'] >= 0
        assert stats['annual_conversion_cap'] > 0
    
    def test_conversion_consistency_validation(self):
        """Test conversion consistency validation."""
        config = SimulationConfig(initial_workers=1000, years=5, seed=42)
        sim = Simulation(config)
        sim.run()
        
        assert sim.validate_conversion_consistency()
    
    def test_temporary_worker_queue_management(self):
        """Test that temporary worker queue is managed correctly."""
        config = SimulationConfig(initial_workers=1000, years=3, seed=42)
        sim = Simulation(config)
        
        initial_queue_size = len(sim.temp_worker_queue)
        
        # Run one step
        sim.step()
        
        # Queue should have new temporary workers minus conversions
        state = sim.states[-1]
        expected_queue_size = (initial_queue_size + 
                             state.new_temporary - state.converted_temps)
        assert len(sim.temp_worker_queue) == expected_queue_size

class TestModels:
    """Test cases for data models."""
    
    def test_worker_creation(self):
        """Test Worker model creation and validation."""
        worker = Worker(id=1, status=WorkerStatus.PERMANENT, age=30)
        
        assert worker.is_permanent
        assert not worker.is_temporary
        assert worker.age == 30
        assert worker.year_joined == 2025  # Default value
    
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
    
    # NEW FOR SPEC-2: Test new model components
    def test_temporary_worker_model(self):
        """Test TemporaryWorker model."""
        temp_worker = TemporaryWorker(worker_id=123, year_joined=2025)
        
        assert temp_worker.worker_id == 123
        assert temp_worker.year_joined == 2025
    
    def test_temporary_worker_ordering(self):
        """Test TemporaryWorker FIFO ordering."""
        worker1 = TemporaryWorker(worker_id=1, year_joined=2024)
        worker2 = TemporaryWorker(worker_id=2, year_joined=2025)
        
        assert worker1 < worker2
        
        # Test sorting
        workers = [worker2, worker1]
        workers.sort()
        assert workers[0] == worker1
        assert workers[1] == worker2
    
    def test_simulation_state_with_conversions(self):
        """Test SimulationState with conversion data."""
        state = SimulationState(
            year=2025,
            total_workers=1000,
            permanent_workers=950,
            temporary_workers=50,
            new_permanent=20,
            new_temporary=5,
            converted_temps=15
        )
        
        assert state.converted_temps == 15
        assert state.total_workers == 1000
        assert state.permanent_workers + state.temporary_workers == state.total_workers

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
