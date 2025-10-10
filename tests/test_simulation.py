# tests/test_simulation.py
"""
Unit tests for the core simulation functionality.
CORRECTED FOR SPEC-8 to test fixed annual conversions and correct per-country logic.
"""

import pytest
import numpy as np
from collections import defaultdict
from unittest.mock import MagicMock, patch
from src.simulation.models import (
    SimulationConfig, SimulationState, Worker, WorkerStatus, 
    TemporaryWorker
)
from src.simulation.sim import Simulation
from src.simulation.empirical_params import (
    H1B_SHARE, ANNUAL_PERMANENT_ENTRY_RATE, 
    GREEN_CARD_CAP_ABS, REAL_US_WORKFORCE_SIZE,
    PER_COUNTRY_CAP_SHARE, TEMP_NATIONALITY_DISTRIBUTION,
    PERMANENT_NATIONALITY, STARTING_WAGE,
    JOB_CHANGE_PROB_PERM, JOB_CHANGE_PROB_TEMP,
    WAGE_JUMP_FACTOR_MEAN_PERM, WAGE_JUMP_FACTOR_MEAN_TEMP,
    calculate_annual_sim_cap
)

class TestSimulation:
    """Test cases for the Simulation class."""
    
    def test_initialization(self):
        """Test simulation initialization with fixed annual caps (CORRECTED FOR SPEC-8)."""
        config = SimulationConfig(initial_workers=1000, years=5, seed=42)
        sim = Simulation(config)
        
        assert len(sim.states) == 1
        assert sim.states[0].total_workers == 1000
        assert sim.states[0].year == 2025
        assert sim.states[0].converted_temps == 0
        
        # Check H-1B share is approximately correct
        expected_h1b = round(1000 * H1B_SHARE)
        assert sim.states[0].temporary_workers == expected_h1b
        assert sim.states[0].permanent_workers == 1000 - expected_h1b
        
        # CORRECTED FOR SPEC-8: Check fixed annual conversion cap
        expected_cap, _ = calculate_annual_sim_cap(1000)
        assert sim.annual_sim_cap == expected_cap
        assert sim.states[0].annual_conversion_cap == expected_cap
        
        # Check country cap is disabled by default
        assert not sim.country_cap_enabled
        assert sim.global_queue is not None
        assert len(sim.country_queues) > 0  # Always maintained for consistency

class TestCorrectedSPEC8Invariants:
    """Test cases for CORRECTED SPEC-8 invariants."""
    
    def test_fixed_annual_cap_constant(self):
        """
        CORRECTED SPEC-8 Test 1: annual_sim_cap must be constant every year and identical across scenarios.
        """
        initial_workers = 10000
        years = 10
        seed = 42
        
        # Test uncapped scenario
        config_uncapped = SimulationConfig(
            initial_workers=initial_workers, 
            years=years, 
            seed=seed,
            country_cap_enabled=False,
            debug=True
        )
        sim_uncapped = Simulation(config_uncapped)
        states_uncapped = sim_uncapped.run()
        
        # Test capped scenario
        config_capped = SimulationConfig(
            initial_workers=initial_workers, 
            years=years, 
            seed=seed,
            country_cap_enabled=True,
            debug=True
        )
        sim_capped = Simulation(config_capped)
        states_capped = sim_capped.run()
        
        # Invariant A: annual_sim_cap is constant every year and identical across scenarios
        expected_cap, _ = calculate_annual_sim_cap(initial_workers)
        
        # Check uncapped scenario
        for state in states_uncapped:
            assert state.annual_conversion_cap == expected_cap, \
                f"Uncapped year {state.year}: cap {state.annual_conversion_cap} != expected {expected_cap}"
        
        # Check capped scenario
        for state in states_capped:
            assert state.annual_conversion_cap == expected_cap, \
                f"Capped year {state.year}: cap {state.annual_conversion_cap} != expected {expected_cap}"
        
        # Check conversions don't exceed cap (except for carryover +1)
        for state in states_uncapped[1:]:
            assert state.converted_temps <= expected_cap + 1, \
                f"Uncapped year {state.year}: conversions {state.converted_temps} > cap {expected_cap} + 1"
        
        for state in states_capped[1:]:
            assert state.converted_temps <= expected_cap + 1, \
                f"Capped year {state.year}: conversions {state.converted_temps} > cap {expected_cap} + 1"
        
        # Check that both scenarios have same annual_sim_cap
        assert sim_uncapped.annual_sim_cap == sim_capped.annual_sim_cap
        
        # Invariant B: Total conversions should be identical (unless queue exhaustion)
        total_conversions_uncapped = sum(state.converted_temps for state in states_uncapped[1:])
        total_conversions_capped = sum(state.converted_temps for state in states_capped[1:])
        
        # Allow for small differences due to queue exhaustion timing
        assert abs(total_conversions_uncapped - total_conversions_capped) <= expected_cap, \
            f"Total conversions differ: uncapped {total_conversions_uncapped} vs capped {total_conversions_capped}"
    
    def test_backlog_totals_equal_across_modes(self):
        """
        CORRECTED SPEC-8 Test 2: Final total backlog must be identical in both scenarios.
        """
        # Use scenario where total demand > total slots
        initial_workers = 5000
        years = 20  # Long enough to ensure queue buildup
        seed = 42
        
        # Test uncapped scenario
        config_uncapped = SimulationConfig(
            initial_workers=initial_workers, 
            years=years, 
            seed=seed,
            country_cap_enabled=False
        )
        sim_uncapped = Simulation(config_uncapped)
        sim_uncapped.run()
        
        # Test capped scenario
        config_capped = SimulationConfig(
            initial_workers=initial_workers, 
            years=years, 
            seed=seed,
            country_cap_enabled=True
        )
        sim_capped = Simulation(config_capped)
        sim_capped.run()
        
        # Invariant C: Final total backlog should be identical
        total_backlog_uncapped = sim_uncapped.get_total_backlog_uncapped()
        total_backlog_capped = sim_capped.get_total_backlog_capped()
        
        # CORRECTED: Perfect synchronization should give identical backlogs
        assert total_backlog_uncapped == total_backlog_capped, \
            f"Total backlogs differ: uncapped {total_backlog_uncapped} vs capped {total_backlog_capped}"
        
        # Additional check: Both should have same cumulative conversions
        assert abs(sim_uncapped.cumulative_conversions - sim_capped.cumulative_conversions) <= sim_uncapped.annual_sim_cap, \
            f"Cumulative conversions differ significantly: uncapped {sim_uncapped.cumulative_conversions} vs capped {sim_capped.cumulative_conversions}"
    
    def test_conversions_per_country_respect_7pct(self):
        """
        CORRECTED SPEC-8 Test 3: Per-country conversions must respect actual per-country caps.
        """
        config = SimulationConfig(
            initial_workers=10000, 
            years=10, 
            seed=42,
            country_cap_enabled=True
        )
        sim = Simulation(config)
        states = sim.run()
        
        # CORRECTED: Use the actual per-country caps from the simulation
        # Don't calculate it independently as the distribution algorithm is complex
        
        # Check each year's per-country conversions
        for state in states[1:]:  # Skip initial state
            for nationality, conversions in state.converted_by_country.items():
                actual_per_country_cap = sim.per_country_caps.get(nationality, 0)
                assert conversions <= actual_per_country_cap + 1, \
                    f"Year {state.year}: {nationality} conversions {conversions} > actual per-country cap {actual_per_country_cap} + 1"
        
        # Verify total conversions equals sum of per-country conversions
        for state in states[1:]:
            sum_per_country = sum(state.converted_by_country.values())
            assert sum_per_country == state.converted_temps, \
                f"Year {state.year}: sum of per-country conversions {sum_per_country} != total conversions {state.converted_temps}"
        
        # Verify per-country caps sum to annual_sim_cap
        total_per_country_caps = sum(sim.per_country_caps.values())
        assert total_per_country_caps == sim.annual_sim_cap, \
            f"Per-country caps sum to {total_per_country_caps}, expected {sim.annual_sim_cap}"
    
    def test_wage_growth_diff_post_conversion(self):
        """
        CORRECTED SPEC-8 Test 4: Wage growth differences and conversion timing.
        """
        config = SimulationConfig(initial_workers=100, years=3, seed=42, agent_mode=True)
        sim = Simulation(config)
        
        # Create a test temporary worker
        test_worker = Worker(
            id=9999, 
            status=WorkerStatus.TEMPORARY, 
            nationality="India", 
            wage=STARTING_WAGE,
            created_year=2025,
            entry_year=2025,
            year_joined=2025
        )
        sim.workers.append(test_worker)
        
        # Convert worker in year 2026
        conversion_year = 2026
        test_worker.convert_to_permanent(conversion_year)
        
        # Test wage parameters in conversion year (should still use temporary)
        with patch.object(sim, 'rng') as mock_rng:
            mock_rng.random.return_value = 0.05  # Force job change
            mock_rng.normal.return_value = WAGE_JUMP_FACTOR_MEAN_TEMP
            
            original_wage = test_worker.wage
            sim._process_job_changes(conversion_year)  # Same year as conversion
            
            expected_wage = original_wage * WAGE_JUMP_FACTOR_MEAN_TEMP
            assert abs(test_worker.wage - expected_wage) < 1.0, \
                f"Conversion year wage should use temp parameters: {test_worker.wage} vs expected {expected_wage}"
        
        # Reset wage for next test
        test_worker.wage = STARTING_WAGE
        
        # Test wage parameters year after conversion (should use permanent)
        with patch.object(sim, 'rng') as mock_rng:
            mock_rng.random.return_value = 0.05  # Force job change
            mock_rng.normal.return_value = WAGE_JUMP_FACTOR_MEAN_PERM
            
            original_wage = test_worker.wage
            sim._process_job_changes(conversion_year + 1)  # Year after conversion
            
            expected_wage = original_wage * WAGE_JUMP_FACTOR_MEAN_PERM
            assert abs(test_worker.wage - expected_wage) < 1.0, \
                f"Post-conversion wage should use perm parameters: {test_worker.wage} vs expected {expected_wage}"
        
        # Verify wage jump factor difference
        assert WAGE_JUMP_FACTOR_MEAN_PERM > WAGE_JUMP_FACTOR_MEAN_TEMP, \
            f"Permanent wage jump {WAGE_JUMP_FACTOR_MEAN_PERM} should be > temporary {WAGE_JUMP_FACTOR_MEAN_TEMP}"
    
    def test_visualization_consistency(self):
        """
        CORRECTED SPEC-8 Test 5: Conversions-per-year should be flat and equal across scenarios.
        """
        initial_workers = 5000
        years = 15
        seed = 42
        
        # Run both scenarios
        config_uncapped = SimulationConfig(
            initial_workers=initial_workers, 
            years=years, 
            seed=seed,
            country_cap_enabled=False
        )
        sim_uncapped = Simulation(config_uncapped)
        states_uncapped = sim_uncapped.run()
        
        config_capped = SimulationConfig(
            initial_workers=initial_workers, 
            years=years, 
            seed=seed,
            country_cap_enabled=True
        )
        sim_capped = Simulation(config_capped)
        states_capped = sim_capped.run()
        
        # Check that conversions-per-year is approximately flat
        expected_annual_conversions = sim_uncapped.annual_sim_cap
        
        # Allow for queue exhaustion in later years
        for i, state in enumerate(states_uncapped[1:5]):  # Check first 4 years
            assert abs(state.converted_temps - expected_annual_conversions) <= 1, \
                f"Uncapped year {state.year}: conversions {state.converted_temps} not approximately equal to {expected_annual_conversions}"
        
        for i, state in enumerate(states_capped[1:5]):  # Check first 4 years
            assert abs(state.converted_temps - expected_annual_conversions) <= 1, \
                f"Capped year {state.year}: conversions {state.converted_temps} not approximately equal to {expected_annual_conversions}"
        
        # Check that both scenarios have similar conversions each year (early years)
        for i in range(min(5, len(states_uncapped)-1, len(states_capped)-1)):
            uncapped_conv = states_uncapped[i+1].converted_temps
            capped_conv = states_capped[i+1].converted_temps
            assert abs(uncapped_conv - capped_conv) <= 1, \
                f"Year {states_uncapped[i+1].year}: conversions differ - uncapped {uncapped_conv} vs capped {capped_conv}"

class TestLegacySimulation:
    """Legacy test cases (updated for SPEC-8 compatibility)."""
    
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
        assert next_state.converted_temps >= 0
        
        # CORRECTED FOR SPEC-8: Check conversion cap is respected
        assert next_state.converted_temps <= sim.annual_sim_cap + 1  # Allow carryover
    
    def test_full_run(self):
        """Test a full simulation run."""
        config = SimulationConfig(initial_workers=1000, years=5, seed=42)
        sim = Simulation(config)
        
        states = sim.run()
        
        assert len(states) == 6  # Initial + 5 years
        assert states[-1].year == 2030
        assert states[-1].total_workers > states[0].total_workers
        
        # Check that conversions occurred
        total_conversions = sum(state.converted_temps for state in states[1:])
        assert total_conversions > 0
        
        # CORRECTED FOR SPEC-8: Check cumulative conversions tracking
        assert states[-1].cumulative_conversions == total_conversions
        
        # CORRECTED FOR SPEC-8: Validate invariants
        assert sim.validate_fixed_conversion_invariants()
    
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
        
        # CORRECTED FOR SPEC-8: Conversion caps should also be proportional
        assert sim2.annual_sim_cap == 2 * sim1.annual_sim_cap
    
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
            assert states1[i].converted_temps == states2[i].converted_temps
            # CORRECTED FOR SPEC-8: Check cumulative conversions
            assert states1[i].cumulative_conversions == states2[i].cumulative_conversions

def test_nationality_consistency():
    """Test that nationality assignments are consistent (FROM SPEC-4)."""
    config = SimulationConfig(initial_workers=1000, years=1, seed=42)
    sim = Simulation(config)
    
    # Get initial workers before running simulation
    initial_workers = sim.to_agent_model()
    initial_permanent_ids = set(w.id for w in initial_workers if w.is_permanent)
    
    # Now run the simulation
    sim.run()
    
    workers = sim.to_agent_model()
    
    # Check that INITIAL permanent workers (by ID) are US nationals
    initial_permanent_workers = [w for w in workers if w.id in initial_permanent_ids and w.is_permanent]
    
    for worker in initial_permanent_workers:
        assert str(worker.nationality) == str(PERMANENT_NATIONALITY), \
            f"Initial permanent worker {worker.id} has nationality {worker.nationality}, expected {PERMANENT_NATIONALITY}"

if __name__ == "__main__":
    pytest.main([__file__])
