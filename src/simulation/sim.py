# src/simulation/sim.py
"""
Core simulation engine for workforce growth modeling.
Implements the Simulation class with step() and run() methods.
Includes temporary-to-permanent conversion logic (SPEC-2).
"""

import math
import logging
from typing import List, Optional, Tuple, Deque
from collections import deque
import numpy as np

from .models import SimulationState, SimulationConfig, Worker, WorkerStatus, TemporaryWorker
from .empirical_params import (
    H1B_SHARE, ANNUAL_PERMANENT_ENTRY_RATE, ANNUAL_H1B_ENTRY_RATE,
    DEFAULT_YEARS, DEFAULT_SEED, GREEN_CARD_CAP_ABS, REAL_US_WORKFORCE_SIZE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Simulation:
    """
    Workforce growth simulation engine.
    
    Operates on aggregate counts for efficiency, with hooks for future
    agent-level simulation capabilities. Includes temporary-to-permanent
    conversion system (SPEC-2).
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize simulation with configuration.
        
        Args:
            config: SimulationConfig object with simulation parameters
        """
        self.config = config
        self.states: List[SimulationState] = []
        self.current_year = 2025  # Base year
        
        # Set up random number generator for reproducibility
        self.rng = np.random.default_rng(config.seed or DEFAULT_SEED)
        
        # NEW FOR SPEC-2: Calculate annual green card conversion cap
        self.annual_conversion_cap = self._calculate_conversion_cap()
        
        # NEW FOR SPEC-2: FIFO queue for temporary workers
        self.temp_worker_queue: Deque[TemporaryWorker] = deque()
        self.next_worker_id = 0
        
        # Initialize simulation state
        self._initialize_workforce()
        
        logger.info(f"Initialized simulation with {config.initial_workers} workers")
        logger.info(f"Initial H-1B share: {self.states[0].h1b_share:.3%}")
        logger.info(f"Annual green card conversion cap: {self.annual_conversion_cap}")
        
    def _calculate_conversion_cap(self) -> int:
        """
        Calculate the annual conversion cap proportional to initial workforce.
        
        Returns:
            Annual number of temporary workers that can convert to permanent
        """
        cap_proportion = GREEN_CARD_CAP_ABS / REAL_US_WORKFORCE_SIZE
        annual_cap = round(self.config.initial_workers * cap_proportion)
        
        logger.info(f"Green card cap calculation: {GREEN_CARD_CAP_ABS:,} / {REAL_US_WORKFORCE_SIZE:,} "
                   f"* {self.config.initial_workers:,} = {annual_cap}")
        
        return max(1, annual_cap)  # Ensure at least 1 conversion possible
        
    def _initialize_workforce(self) -> None:
        """Initialize the workforce with proper H-1B/permanent split."""
        initial_temporary = round(self.config.initial_workers * H1B_SHARE)
        initial_permanent = self.config.initial_workers - initial_temporary
        
        # Handle edge case for very small initial populations
        if self.config.initial_workers < 100:
            logger.warning(
                f"Small initial population ({self.config.initial_workers}). "
                "Results may show discretization effects."
            )
        
        # NEW FOR SPEC-2: Initialize temporary worker queue
        for i in range(initial_temporary):
            temp_worker = TemporaryWorker(
                worker_id=self.next_worker_id,
                year_joined=self.current_year
            )
            self.temp_worker_queue.append(temp_worker)
            self.next_worker_id += 1
        
        initial_state = SimulationState(
            year=self.current_year,
            total_workers=self.config.initial_workers,
            permanent_workers=initial_permanent,
            temporary_workers=initial_temporary,
            new_permanent=0,
            new_temporary=0,
            converted_temps=0
        )
        
        self.states.append(initial_state)
        
    def step(self) -> SimulationState:
        """
        Execute one simulation time step (one year).
        
        Returns:
            SimulationState for the new year
        """
        if not self.states:
            raise RuntimeError("Simulation not initialized")
            
        current_state = self.states[-1]
        next_year = current_state.year + 1
        
        # Calculate new entrants based on current total workforce
        # Using round() for consistent rounding strategy as specified
        new_permanent = round(current_state.total_workers * ANNUAL_PERMANENT_ENTRY_RATE)
        new_temporary = round(current_state.total_workers * ANNUAL_H1B_ENTRY_RATE)
        
        # NEW FOR SPEC-2: Add new temporary workers to the queue
        for i in range(new_temporary):
            temp_worker = TemporaryWorker(
                worker_id=self.next_worker_id,
                year_joined=next_year
            )
            self.temp_worker_queue.append(temp_worker)
            self.next_worker_id += 1
        
        # NEW FOR SPEC-2: Process temporary-to-permanent conversions (FIFO)
        converted_temps = min(len(self.temp_worker_queue), self.annual_conversion_cap)
        
        # Remove converted workers from queue
        for _ in range(converted_temps):
            if self.temp_worker_queue:
                self.temp_worker_queue.popleft()
        
        # Update workforce counts
        permanent_workers = current_state.permanent_workers + new_permanent + converted_temps
        temporary_workers = current_state.temporary_workers + new_temporary - converted_temps
        total_workers = permanent_workers + temporary_workers
        
        next_state = SimulationState(
            year=next_year,
            total_workers=total_workers,
            permanent_workers=permanent_workers,
            temporary_workers=temporary_workers,
            new_permanent=new_permanent,
            new_temporary=new_temporary,
            converted_temps=converted_temps
        )
        
        self.states.append(next_state)
        
        logger.debug(f"Year {next_year}: Total={total_workers}, "
                    f"Permanent={permanent_workers}, Temporary={temporary_workers}, "
                    f"Converted={converted_temps}")
        
        return next_state
    
    def run(self) -> List[SimulationState]:
        """
        Run the complete simulation for the configured number of years.
        
        Returns:
            List of SimulationState objects for each year
        """
        logger.info(f"Running simulation for {self.config.years} years")
        
        for year in range(self.config.years):
            self.step()
            
            # Progress logging every 5 years
            if (year + 1) % 5 == 0:
                current_state = self.states[-1]
                logger.info(f"Year {current_state.year}: "
                           f"{current_state.total_workers:,} total workers "
                           f"({current_state.h1b_share:.2%} H-1B), "
                           f"{current_state.converted_temps} conversions")
        
        final_state = self.states[-1]
        total_conversions = sum(state.converted_temps for state in self.states[1:])
        logger.info(f"Simulation complete. Final workforce: {final_state.total_workers:,} "
                   f"({final_state.h1b_share:.2%} H-1B), "
                   f"{total_conversions} total conversions")
        
        return self.states
    
    def get_growth_rate(self, year_span: int = 1) -> Optional[float]:
        """
        Calculate annualized growth rate over specified time span.
        
        Args:
            year_span: Number of years to look back for growth calculation
            
        Returns:
            Annualized growth rate as decimal, or None if insufficient data
        """
        if len(self.states) < year_span + 1:
            return None
            
        current_total = self.states[-1].total_workers
        past_total = self.states[-(year_span + 1)].total_workers
        
        if past_total == 0:
            return None
            
        return (current_total / past_total) ** (1 / year_span) - 1
    
    def validate_proportional_growth(self, tolerance: float = 0.01) -> bool:
        """
        Validate that workforce growth maintains proportional relationships.
        
        Args:
            tolerance: Acceptable deviation from expected proportions
            
        Returns:
            True if growth is proportionally consistent
        """
        if len(self.states) < 2:
            return True
            
        initial_state = self.states[0]
        final_state = self.states[-1]
        
        # Check if H-1B share has remained relatively stable
        # (it should change slightly due to different growth rates and conversions)
        share_change = abs(final_state.h1b_share - initial_state.h1b_share)
        
        # Expected share change based on different growth rates and conversions
        expected_share_change = abs(ANNUAL_H1B_ENTRY_RATE - ANNUAL_PERMANENT_ENTRY_RATE) * len(self.states)
        
        return share_change <= expected_share_change + tolerance
    
    def validate_conversion_consistency(self) -> bool:
        """
        NEW FOR SPEC-2: Validate that conversions don't exceed cap and maintain consistency.
        
        Returns:
            True if conversion logic is consistent
        """
        for state in self.states[1:]:  # Skip initial state
            # Check conversion cap is never exceeded
            if state.converted_temps > self.annual_conversion_cap:
                logger.error(f"Year {state.year}: Conversions {state.converted_temps} "
                           f"exceed cap {self.annual_conversion_cap}")
                return False
                
            # Check total workers remain consistent
            prev_state = self.states[self.states.index(state) - 1]
            expected_total = (prev_state.total_workers + 
                            state.new_permanent + state.new_temporary)
            
            if state.total_workers != expected_total:
                logger.error(f"Year {state.year}: Total workers inconsistent. "
                           f"Expected {expected_total}, got {state.total_workers}")
                return False
        
        return True
    
    def to_agent_model(self) -> List[Worker]:
        """
        Convert current aggregate simulation to individual Worker agents.
        
        This is a hook for future agent-based modeling capabilities.
        Currently creates representative agents based on aggregate counts.
        
        Returns:
            List of Worker objects representing current workforce
        """
        if not self.states:
            return []
            
        current_state = self.states[-1]
        workers = []
        
        # Create permanent workers
        for i in range(current_state.permanent_workers):
            worker = Worker(
                id=i,
                status=WorkerStatus.PERMANENT,
                age=self.rng.integers(25, 65),  # Random age in working range
                entry_year=self.rng.integers(2000, current_state.year + 1)
            )
            workers.append(worker)
        
        # Create temporary/H-1B workers using queue data
        temp_worker_list = list(self.temp_worker_queue)
        for i, temp_worker in enumerate(temp_worker_list):
            worker = Worker(
                id=temp_worker.worker_id,
                status=WorkerStatus.TEMPORARY,
                age=self.rng.integers(25, 55),  # H-1B workers tend to be younger
                entry_year=temp_worker.year_joined,
                year_joined=temp_worker.year_joined
            )
            workers.append(worker)
            
        logger.info(f"Created {len(workers)} worker agents from aggregate model")
        return workers
    
    def get_summary_stats(self) -> dict:
        """
        Get summary statistics for the simulation run.
        
        Returns:
            Dictionary with key simulation metrics
        """
        if not self.states:
            return {}
            
        initial_state = self.states[0]
        final_state = self.states[-1]
        
        total_growth = final_state.total_workers - initial_state.total_workers
        years_simulated = len(self.states) - 1
        total_conversions = sum(state.converted_temps for state in self.states[1:])
        
        return {
            "years_simulated": years_simulated,
            "initial_workforce": initial_state.total_workers,
            "final_workforce": final_state.total_workers,
            "total_growth": total_growth,
            "average_annual_growth_rate": self.get_growth_rate(years_simulated) or 0,
            "initial_h1b_share": initial_state.h1b_share,
            "final_h1b_share": final_state.h1b_share,
            "h1b_share_change": final_state.h1b_share - initial_state.h1b_share,
            "total_new_permanent": sum(state.new_permanent for state in self.states[1:]),
            "total_new_temporary": sum(state.new_temporary for state in self.states[1:]),
            "total_conversions": total_conversions,  # NEW FOR SPEC-2
            "annual_conversion_cap": self.annual_conversion_cap,  # NEW FOR SPEC-2
            "conversion_utilization": total_conversions / (years_simulated * self.annual_conversion_cap) if years_simulated > 0 else 0  # NEW FOR SPEC-2
        }
