# src/simulation/sim.py
"""
Core simulation engine for workforce growth modeling.
Implements the Simulation class with step() and run() methods.
Includes temporary-to-permanent conversion logic (SPEC-2) and
individual wage tracking with job-to-job transitions (SPEC-3).
"""

import math
import logging
from typing import List, Optional, Tuple, Deque, Dict
from collections import deque
import numpy as np

from .models import (
    SimulationState, SimulationConfig, Worker, WorkerStatus, 
    TemporaryWorker, WageStatistics
)
from .empirical_params import (
    H1B_SHARE, ANNUAL_PERMANENT_ENTRY_RATE, ANNUAL_H1B_ENTRY_RATE,
    DEFAULT_YEARS, DEFAULT_SEED, GREEN_CARD_CAP_ABS, REAL_US_WORKFORCE_SIZE,
    STARTING_WAGE, JOB_CHANGE_PROB_PERM, TEMP_JOB_CHANGE_PENALTY,
    WAGE_JUMP_FACTOR_MEAN, WAGE_JUMP_FACTOR_STD, INDUSTRY_NAME
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Simulation:
    """
    Workforce growth simulation engine.
    
    Updated for SPEC-3 to operate in agent-mode by default with individual
    wage tracking and job-to-job transition mechanics. Includes temporary-to-permanent
    conversion system from SPEC-2.
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
        
        # Calculate annual green card conversion cap (FROM SPEC-2)
        self.annual_conversion_cap = self._calculate_conversion_cap()
        
        # FIFO queue for temporary workers (FROM SPEC-2)
        self.temp_worker_queue: Deque[TemporaryWorker] = deque()
        
        # NEW FOR SPEC-3: Agent-mode worker tracking
        self.workers: List[Worker] = []
        self.next_worker_id = 0
        
        # Calculate job change probabilities (NEW FOR SPEC-3)
        self.job_change_prob_perm = JOB_CHANGE_PROB_PERM
        self.job_change_prob_temp = JOB_CHANGE_PROB_PERM * (1 - TEMP_JOB_CHANGE_PENALTY)
        
        # Initialize simulation state
        self._initialize_workforce()
        
        logger.info(f"Initialized simulation with {config.initial_workers} workers in {INDUSTRY_NAME}")
        logger.info(f"Initial H-1B share: {self.states[0].h1b_share:.3%}")
        logger.info(f"Annual green card conversion cap: {self.annual_conversion_cap}")
        logger.info(f"Job change probabilities - Permanent: {self.job_change_prob_perm:.2%}, "
                   f"Temporary: {self.job_change_prob_temp:.2%}")
        
        if config.agent_mode and config.initial_workers > 200000:
            logger.warning(f"Agent-mode with {config.initial_workers:,} workers will be slow. "
                          "Consider using --count-mode for large simulations.")
        
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
        """Initialize the workforce with proper H-1B/permanent split and starting wages."""
        initial_temporary = round(self.config.initial_workers * H1B_SHARE)
        initial_permanent = self.config.initial_workers - initial_temporary
        
        # Handle edge case for very small initial populations
        if self.config.initial_workers < 100:
            logger.warning(
                f"Small initial population ({self.config.initial_workers}). "
                "Results may show discretization effects."
            )
        
        # NEW FOR SPEC-3: Create individual Worker objects
        if self.config.agent_mode:
            # Create permanent workers
            for i in range(initial_permanent):
                worker = Worker(
                    id=self.next_worker_id,
                    status=WorkerStatus.PERMANENT,
                    age=self.rng.integers(25, 65),
                    wage=STARTING_WAGE,
                    created_year=self.current_year,
                    entry_year=self.current_year,
                    year_joined=self.current_year
                )
                self.workers.append(worker)
                self.next_worker_id += 1
            
            # Create temporary workers and add to FIFO queue
            for i in range(initial_temporary):
                worker = Worker(
                    id=self.next_worker_id,
                    status=WorkerStatus.TEMPORARY,
                    age=self.rng.integers(25, 55),
                    wage=STARTING_WAGE,
                    created_year=self.current_year,
                    entry_year=self.current_year,
                    year_joined=self.current_year
                )
                self.workers.append(worker)
                
                # Add to conversion queue
                temp_worker = TemporaryWorker(
                    worker_id=worker.id,
                    year_joined=self.current_year
                )
                self.temp_worker_queue.append(temp_worker)
                self.next_worker_id += 1
            
            # Calculate initial wage statistics
            wage_stats = WageStatistics.calculate(self.workers)
        else:
            # Count-mode: approximate wage statistics
            wage_stats = WageStatistics(
                total_workers=self.config.initial_workers,
                permanent_workers=initial_permanent,
                temporary_workers=initial_temporary,
                avg_wage_total=STARTING_WAGE,
                avg_wage_permanent=STARTING_WAGE,
                avg_wage_temporary=STARTING_WAGE,
                total_wage_bill=STARTING_WAGE * self.config.initial_workers
            )
        
        initial_state = SimulationState(
            year=self.current_year,
            total_workers=self.config.initial_workers,
            permanent_workers=initial_permanent,
            temporary_workers=initial_temporary,
            new_permanent=0,
            new_temporary=0,
            converted_temps=0,
            avg_wage_total=wage_stats.avg_wage_total,
            avg_wage_permanent=wage_stats.avg_wage_permanent,
            avg_wage_temporary=wage_stats.avg_wage_temporary,
            total_wage_bill=wage_stats.total_wage_bill
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
        new_permanent = round(current_state.total_workers * ANNUAL_PERMANENT_ENTRY_RATE)
        new_temporary = round(current_state.total_workers * ANNUAL_H1B_ENTRY_RATE)
        
        if self.config.agent_mode:
            # Agent-mode: work with individual Worker objects
            converted_temps = self._process_agent_mode_step(next_year, new_permanent, new_temporary)
            wage_stats = WageStatistics.calculate(self.workers)
            
            permanent_count = len([w for w in self.workers if w.is_permanent])
            temporary_count = len([w for w in self.workers if w.is_temporary])
        else:
            # Count-mode: approximate using compartmental means
            converted_temps = self._process_count_mode_step(next_year, new_permanent, new_temporary)
            wage_stats = self._calculate_count_mode_wages(current_state, new_permanent, new_temporary, converted_temps)
            
            permanent_count = current_state.permanent_workers + new_permanent + converted_temps
            temporary_count = current_state.temporary_workers + new_temporary - converted_temps
        
        total_workers = permanent_count + temporary_count
        
        next_state = SimulationState(
            year=next_year,
            total_workers=total_workers,
            permanent_workers=permanent_count,
            temporary_workers=temporary_count,
            new_permanent=new_permanent,
            new_temporary=new_temporary,
            converted_temps=converted_temps,
            avg_wage_total=wage_stats.avg_wage_total,
            avg_wage_permanent=wage_stats.avg_wage_permanent,
            avg_wage_temporary=wage_stats.avg_wage_temporary,
            total_wage_bill=wage_stats.total_wage_bill
        )
        
        self.states.append(next_state)
        
        logger.debug(f"Year {next_year}: Total={total_workers}, "
                    f"Permanent={permanent_count}, Temporary={temporary_count}, "
                    f"Converted={converted_temps}, Avg Wage=${wage_stats.avg_wage_total:,.0f}")
        
        return next_state
    
    def _process_agent_mode_step(self, next_year: int, new_permanent: int, new_temporary: int) -> int:
        """
        Process one simulation step in agent-mode.
        
        Args:
            next_year: The year being simulated
            new_permanent: Number of new permanent workers to add
            new_temporary: Number of new temporary workers to add
            
        Returns:
            Number of temporary workers converted to permanent
        """
        # 1. Add new permanent workers
        for _ in range(new_permanent):
            worker = Worker(
                id=self.next_worker_id,
                status=WorkerStatus.PERMANENT,
                age=self.rng.integers(25, 65),
                wage=STARTING_WAGE,
                created_year=next_year,
                entry_year=next_year,
                year_joined=next_year
            )
            self.workers.append(worker)
            self.next_worker_id += 1
        
        # 2. Add new temporary workers
        for _ in range(new_temporary):
            worker = Worker(
                id=self.next_worker_id,
                status=WorkerStatus.TEMPORARY,
                age=self.rng.integers(25, 55),
                wage=STARTING_WAGE,
                created_year=next_year,
                entry_year=next_year,
                year_joined=next_year
            )
            self.workers.append(worker)
            
            # Add to conversion queue
            temp_worker = TemporaryWorker(
                worker_id=worker.id,
                year_joined=next_year
            )
            self.temp_worker_queue.append(temp_worker)
            self.next_worker_id += 1
        
        # 3. Process job changes and wage updates
        self._process_job_changes()
        
        # 4. Process temporary-to-permanent conversions (FIFO)
        converted_temps = min(len(self.temp_worker_queue), self.annual_conversion_cap)
        converted_worker_ids = set()
        
        for _ in range(converted_temps):
            if self.temp_worker_queue:
                temp_worker = self.temp_worker_queue.popleft()
                converted_worker_ids.add(temp_worker.worker_id)
        
        # Update worker status for converted workers
        for worker in self.workers:
            if worker.id in converted_worker_ids:
                worker.status = WorkerStatus.PERMANENT
        
        return converted_temps
    
    def _process_job_changes(self) -> None:
        """
        Process job changes and wage updates for all workers (NEW FOR SPEC-3).
        """
        for worker in self.workers:
            # Determine job change probability based on status
            if worker.is_permanent:
                job_change_prob = self.job_change_prob_perm
            else:
                job_change_prob = self.job_change_prob_temp
            
            # Sample job change event
            if self.rng.random() < job_change_prob:
                # Sample wage jump factor
                wage_jump = self.rng.normal(WAGE_JUMP_FACTOR_MEAN, WAGE_JUMP_FACTOR_STD)
                worker.apply_wage_jump(wage_jump)
    
    def _process_count_mode_step(self, next_year: int, new_permanent: int, new_temporary: int) -> int:
        """
        Process one simulation step in count-mode (approximation).
        
        Args:
            next_year: The year being simulated  
            new_permanent: Number of new permanent workers to add
            new_temporary: Number of new temporary workers to add
            
        Returns:
            Number of temporary workers converted to permanent
        """
        # Simple queue size tracking for count-mode
        current_queue_size = len(self.temp_worker_queue)
        
        # Add new temporary workers to queue (approximated)
        for i in range(new_temporary):
            temp_worker = TemporaryWorker(
                worker_id=self.next_worker_id + i,
                year_joined=next_year
            )
            self.temp_worker_queue.append(temp_worker)
        
        self.next_worker_id += new_temporary
        
        # Process conversions
        converted_temps = min(len(self.temp_worker_queue), self.annual_conversion_cap)
        
        for _ in range(converted_temps):
            if self.temp_worker_queue:
                self.temp_worker_queue.popleft()
        
        return converted_temps
    
    def _calculate_count_mode_wages(self, current_state: SimulationState, 
                                  new_permanent: int, new_temporary: int, 
                                  converted_temps: int) -> WageStatistics:
        """
        Calculate approximate wage statistics for count-mode.
        
        Returns:
            WageStatistics with approximated values
        """
        # Approximate wage growth using expected job change rates
        expected_perm_changes = current_state.permanent_workers * self.job_change_prob_perm
        expected_temp_changes = current_state.temporary_workers * self.job_change_prob_temp
        
        # Approximate new wages (simplified)
        wage_growth_factor = (
            (expected_perm_changes + expected_temp_changes) * WAGE_JUMP_FACTOR_MEAN / 
            current_state.total_workers + 1.0
        )
        
        # Update average wages
        new_avg_wage = current_state.avg_wage_total * wage_growth_factor
        
        # Calculate new totals
        new_permanent_total = current_state.permanent_workers + new_permanent + converted_temps
        new_temporary_total = current_state.temporary_workers + new_temporary - converted_temps
        new_total_workers = new_permanent_total + new_temporary_total
        
        # Approximate wage bill
        total_wage_bill = (
            new_avg_wage * current_state.total_workers +  # Existing workers
            STARTING_WAGE * (new_permanent + new_temporary)  # New workers
        )
        
        avg_wage_total = total_wage_bill / new_total_workers if new_total_workers > 0 else 0.0
        
        return WageStatistics(
            total_workers=new_total_workers,
            permanent_workers=new_permanent_total,
            temporary_workers=new_temporary_total,
            avg_wage_total=avg_wage_total,
            avg_wage_permanent=avg_wage_total * 1.02,  # Approximation
            avg_wage_temporary=avg_wage_total * 0.98,  # Approximation
            total_wage_bill=total_wage_bill
        )
    
    def run(self) -> List[SimulationState]:
        """
        Run the complete simulation for the configured number of years.
        
        Returns:
            List of SimulationState objects for each year
        """
        logger.info(f"Running simulation for {self.config.years} years")
        logger.info(f"Mode: {'Agent-based' if self.config.agent_mode else 'Count-based'}")
        
        for year in range(self.config.years):
            self.step()
            
            # Progress logging every 5 years
            if (year + 1) % 5 == 0:
                current_state = self.states[-1]
                logger.info(f"Year {current_state.year}: "
                           f"{current_state.total_workers:,} total workers "
                           f"({current_state.h1b_share:.2%} H-1B), "
                           f"{current_state.converted_temps} conversions, "
                           f"Avg wage: ${current_state.avg_wage_total:,.0f}")
        
        final_state = self.states[-1]
        total_conversions = sum(state.converted_temps for state in self.states[1:])
        logger.info(f"Simulation complete. Final workforce: {final_state.total_workers:,} "
                   f"({final_state.h1b_share:.2%} H-1B), "
                   f"{total_conversions} total conversions, "
                   f"Final avg wage: ${final_state.avg_wage_total:,.0f}")
        
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
    
    def get_wage_growth_rate(self, year_span: int = 1) -> Optional[float]:
        """
        Calculate annualized wage growth rate (NEW FOR SPEC-3).
        
        Args:
            year_span: Number of years to look back for wage growth calculation
            
        Returns:
            Annualized wage growth rate as decimal, or None if insufficient data
        """
        if len(self.states) < year_span + 1:
            return None
            
        current_wage = self.states[-1].avg_wage_total
        past_wage = self.states[-(year_span + 1)].avg_wage_total
        
        if past_wage == 0:
            return None
            
        return (current_wage / past_wage) ** (1 / year_span) - 1
    
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
        share_change = abs(final_state.h1b_share - initial_state.h1b_share)
        
        # Expected share change based on different growth rates and conversions
        expected_share_change = abs(ANNUAL_H1B_ENTRY_RATE - ANNUAL_PERMANENT_ENTRY_RATE) * len(self.states)
        
        return share_change <= expected_share_change + tolerance
    
    def validate_conversion_consistency(self) -> bool:
        """
        Validate that conversions don't exceed cap and maintain consistency.
        
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
    
    def validate_wage_consistency(self) -> bool:
        """
        Validate wage calculations and consistency (NEW FOR SPEC-3).
        
        Returns:
            True if wage calculations are consistent
        """
        for state in self.states:
            # Check that wage statistics are non-negative
            if (state.avg_wage_total < 0 or state.avg_wage_permanent < 0 or 
                state.avg_wage_temporary < 0 or state.total_wage_bill < 0):
                logger.error(f"Year {state.year}: Negative wage statistics detected")
                return False
            
            # Check wage bill consistency
            if state.total_workers > 0:
                expected_wage_bill = state.avg_wage_total * state.total_workers
                if abs(state.total_wage_bill - expected_wage_bill) / expected_wage_bill > 0.01:
                    logger.error(f"Year {state.year}: Wage bill inconsistent with average wage")
                    return False
        
        return True
    
    def to_agent_model(self) -> List[Worker]:
        """
        Return current worker agents (updated for SPEC-3).
        
        Returns:
            List of Worker objects representing current workforce
        """
        if self.config.agent_mode:
            return self.workers.copy()
        else:
            logger.warning("Cannot return individual agents in count-mode")
            return []
    
    def get_summary_stats(self) -> dict:
        """
        Get summary statistics for the simulation run.
        Updated for SPEC-3 to include wage statistics.
        
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
        
        # Wage growth statistics (NEW FOR SPEC-3)
        wage_growth = final_state.avg_wage_total - initial_state.avg_wage_total
        wage_growth_rate = self.get_wage_growth_rate(years_simulated) or 0
        
        return {
            "years_simulated": years_simulated,
            "simulation_mode": "agent" if self.config.agent_mode else "count",
            "industry": INDUSTRY_NAME,
            "initial_workforce": initial_state.total_workers,
            "final_workforce": final_state.total_workers,
            "total_growth": total_growth,
            "average_annual_growth_rate": self.get_growth_rate(years_simulated) or 0,
            "initial_h1b_share": initial_state.h1b_share,
            "final_h1b_share": final_state.h1b_share,
            "h1b_share_change": final_state.h1b_share - initial_state.h1b_share,
            "total_new_permanent": sum(state.new_permanent for state in self.states[1:]),
            "total_new_temporary": sum(state.new_temporary for state in self.states[1:]),
            "total_conversions": total_conversions,
            "annual_conversion_cap": self.annual_conversion_cap,
            "conversion_utilization": total_conversions / (years_simulated * self.annual_conversion_cap) if years_simulated > 0 else 0,
            # NEW FOR SPEC-3: Wage statistics
            "initial_avg_wage": initial_state.avg_wage_total,
            "final_avg_wage": final_state.avg_wage_total,
            "total_wage_growth": wage_growth,
            "average_annual_wage_growth_rate": wage_growth_rate,
            "final_wage_bill": final_state.total_wage_bill,
            "job_change_prob_permanent": self.job_change_prob_perm,
            "job_change_prob_temporary": self.job_change_prob_temp
        }
