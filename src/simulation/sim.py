# src/simulation/sim.py
"""
Core simulation engine for workforce growth modeling.
Implements the Simulation class with step() and run() methods.
Includes temporary-to-permanent conversion logic (SPEC-2),
individual wage tracking with job-to-job transitions (SPEC-3),
nationality segmentation (SPEC-4), and per-country cap system (SPEC-5).
"""

import math
import logging
from typing import List, Optional, Tuple, Deque, Dict
from collections import deque, defaultdict
import numpy as np

from .models import (
    SimulationState, SimulationConfig, Worker, WorkerStatus, 
    TemporaryWorker, WageStatistics, NationalityStatistics, CountryCapStatistics
)
from .empirical_params import (
    H1B_SHARE, ANNUAL_PERMANENT_ENTRY_RATE, ANNUAL_H1B_ENTRY_RATE,
    DEFAULT_YEARS, DEFAULT_SEED, GREEN_CARD_CAP_ABS, REAL_US_WORKFORCE_SIZE,
    STARTING_WAGE, JOB_CHANGE_PROB_PERM, TEMP_JOB_CHANGE_PENALTY,
    WAGE_JUMP_FACTOR_MEAN, WAGE_JUMP_FACTOR_STD, INDUSTRY_NAME,
    TEMP_NATIONALITY_DISTRIBUTION, PERMANENT_NATIONALITY,
    PER_COUNTRY_CAP_SHARE, ENABLE_COUNTRY_CAP
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Simulation:
    """
    Workforce growth simulation engine.
    
    Updated for SPEC-5 to include per-country cap system alongside
    nationality segmentation (SPEC-4), agent-mode wage tracking (SPEC-3), 
    and temporary-to-permanent conversion system (SPEC-2).
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
        
        # NEW FOR SPEC-5: Per-country cap system
        self.country_cap_enabled = config.country_cap_enabled
        self.per_country_cap = round(self.annual_conversion_cap * PER_COUNTRY_CAP_SHARE)
        
        # Conversion queues - either single global or per-country (NEW FOR SPEC-5)
        if self.country_cap_enabled:
            self.country_queues: Dict[str, Deque[TemporaryWorker]] = defaultdict(deque)
            self.global_queue: Optional[Deque[TemporaryWorker]] = None
        else:
            self.global_queue: Deque[TemporaryWorker] = deque()
            self.country_queues: Dict[str, Deque[TemporaryWorker]] = {}
        
        # Agent-mode worker tracking (FROM SPEC-3)
        self.workers: List[Worker] = []
        self.next_worker_id = 0
        
        # Calculate job change probabilities (FROM SPEC-3)
        self.job_change_prob_perm = JOB_CHANGE_PROB_PERM
        self.job_change_prob_temp = JOB_CHANGE_PROB_PERM * (1 - TEMP_JOB_CHANGE_PENALTY)
        
        # Nationality distribution setup (FROM SPEC-4)
        self.temp_nationality_distribution = TEMP_NATIONALITY_DISTRIBUTION.copy()
        self._validate_nationality_distribution()
        
        # Initialize simulation state
        self._initialize_workforce()
        
        logger.info(f"Initialized simulation with {config.initial_workers} workers in {INDUSTRY_NAME}")
        logger.info(f"Initial H-1B share: {self.states[0].h1b_share:.3%}")
        logger.info(f"Annual green card conversion cap: {self.annual_conversion_cap}")
        
        # NEW FOR SPEC-5: Log per-country cap settings
        if self.country_cap_enabled:
            logger.info(f"Per-country cap ENABLED: {self.per_country_cap} conversions per country per year")
            logger.info(f"Per-country cap rate: {PER_COUNTRY_CAP_SHARE:.1%}")
        else:
            logger.info("Per-country cap DISABLED: using global FIFO queue")
        
        logger.info(f"Job change probabilities - Permanent: {self.job_change_prob_perm:.2%}, "
                   f"Temporary: {self.job_change_prob_temp:.2%}")
        
        if config.agent_mode and config.initial_workers > 200000:
            logger.warning(f"Agent-mode with {config.initial_workers:,} workers will be slow. "
                          "Consider using --count-mode for large simulations.")
        
    def _validate_nationality_distribution(self) -> None:
        """Validate that nationality distribution sums to 1.0."""
        total = sum(self.temp_nationality_distribution.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(f"Nationality distribution sum: {total:.6f}, normalizing to 1.0")
            # Normalize the distribution
            for nationality in self.temp_nationality_distribution:
                self.temp_nationality_distribution[nationality] /= total
        
        logger.info(f"Nationality distribution validated. Top nationalities: "
                   f"{dict(list(sorted(self.temp_nationality_distribution.items(), key=lambda x: x[1], reverse=True))[:3])}")
        
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
    
    def _sample_nationality(self, is_temporary: bool) -> str:
        """
        Sample a nationality for a new worker (FROM SPEC-4).
        
        Args:
            is_temporary: True if worker is temporary (H-1B), False if permanent
            
        Returns:
            Nationality string
        """
        if not is_temporary:
            return PERMANENT_NATIONALITY
        
        # Sample from temporary worker nationality distribution
        nationalities = list(self.temp_nationality_distribution.keys())
        probabilities = list(self.temp_nationality_distribution.values())
        
        return self.rng.choice(nationalities, p=probabilities)
    
    def _add_to_conversion_queue(self, worker_id: int, nationality: str, year_joined: int) -> None:
        """
        Add a temporary worker to the appropriate conversion queue (NEW FOR SPEC-5).
        
        Args:
            worker_id: ID of the worker
            nationality: Worker's nationality
            year_joined: Year worker joined as temporary
        """
        temp_worker = TemporaryWorker(
            worker_id=worker_id,
            year_joined=year_joined,
            nationality=nationality
        )
        
        if self.country_cap_enabled:
            # Add to nationality-specific queue
            self.country_queues[nationality].append(temp_worker)
        else:
            # Add to global queue
            self.global_queue.append(temp_worker)
        
    def _initialize_workforce(self) -> None:
        """Initialize the workforce with proper H-1B/permanent split, starting wages, and nationalities."""
        initial_temporary = round(self.config.initial_workers * H1B_SHARE)
        initial_permanent = self.config.initial_workers - initial_temporary
        
        # Handle edge case for very small initial populations
        if self.config.initial_workers < 100:
            logger.warning(
                f"Small initial population ({self.config.initial_workers}). "
                "Results may show discretization effects."
            )
        
        # Agent-mode: Create individual Worker objects (FROM SPEC-3)
        if self.config.agent_mode:
            # Create permanent workers (all U.S. nationals per SPEC-4)
            for i in range(initial_permanent):
                worker = Worker(
                    id=self.next_worker_id,
                    status=WorkerStatus.PERMANENT,
                    nationality=PERMANENT_NATIONALITY,  # FROM SPEC-4
                    age=self.rng.integers(25, 65),
                    wage=STARTING_WAGE,
                    created_year=self.current_year,
                    entry_year=self.current_year,
                    year_joined=self.current_year
                )
                self.workers.append(worker)
                self.next_worker_id += 1
            
            # Create temporary workers with nationality distribution (FROM SPEC-4)
            for i in range(initial_temporary):
                nationality = self._sample_nationality(is_temporary=True)
                worker = Worker(
                    id=self.next_worker_id,
                    status=WorkerStatus.TEMPORARY,
                    nationality=nationality,  # FROM SPEC-4
                    age=self.rng.integers(25, 55),
                    wage=STARTING_WAGE,
                    created_year=self.current_year,
                    entry_year=self.current_year,
                    year_joined=self.current_year
                )
                self.workers.append(worker)
                
                # Add to appropriate conversion queue (NEW FOR SPEC-5)
                self._add_to_conversion_queue(worker.id, nationality, self.current_year)
                self.next_worker_id += 1
            
            # Calculate initial statistics
            wage_stats = WageStatistics.calculate(self.workers)
            nationality_stats = NationalityStatistics.calculate(self.workers)
        else:
            # Count-mode: approximate statistics
            wage_stats = WageStatistics(
                total_workers=self.config.initial_workers,
                permanent_workers=initial_permanent,
                temporary_workers=initial_temporary,
                avg_wage_total=STARTING_WAGE,
                avg_wage_permanent=STARTING_WAGE,
                avg_wage_temporary=STARTING_WAGE,
                total_wage_bill=STARTING_WAGE * self.config.initial_workers
            )
            nationality_stats = NationalityStatistics(
                total_workers=self.config.initial_workers,
                permanent_nationalities={PERMANENT_NATIONALITY: initial_permanent},
                temporary_nationalities={k: round(v * initial_temporary) for k, v in self.temp_nationality_distribution.items()}
            )
        
        # Initialize per-country conversion statistics (NEW FOR SPEC-5)
        initial_conversions = {}
        initial_backlogs = {}
        if self.country_cap_enabled:
            initial_backlogs = {nationality: len(queue) for nationality, queue in self.country_queues.items()}
        elif self.global_queue is not None:
            # For uncapped mode, we can track backlogs by nationality for reporting
            temp_workers_by_nationality = defaultdict(int)
            for worker in (self.workers if self.config.agent_mode else []):
                if worker.is_temporary:
                    temp_workers_by_nationality[worker.nationality] += 1
            initial_backlogs = dict(temp_workers_by_nationality)
        
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
            total_wage_bill=wage_stats.total_wage_bill,
            top_temp_nationalities=nationality_stats.get_top_temporary_nationalities(),  # FROM SPEC-4
            converted_by_country=initial_conversions,  # NEW FOR SPEC-5
            queue_backlog_by_country=initial_backlogs,  # NEW FOR SPEC-5
            country_cap_enabled=self.country_cap_enabled  # NEW FOR SPEC-5
        )
        
        self.states.append(initial_state)
        
        # Print nationality summary if requested (FROM SPEC-4)
        if self.config.show_nationality_summary:
            self._print_nationality_summary("INITIAL", nationality_stats)
        
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
            converted_temps, conversions_by_country = self._process_agent_mode_step(next_year, new_permanent, new_temporary)
            wage_stats = WageStatistics.calculate(self.workers)
            nationality_stats = NationalityStatistics.calculate(self.workers)  # FROM SPEC-4
            
            permanent_count = len([w for w in self.workers if w.is_permanent])
            temporary_count = len([w for w in self.workers if w.is_temporary])
        else:
            # Count-mode: approximate using compartmental means
            converted_temps, conversions_by_country = self._process_count_mode_step(next_year, new_permanent, new_temporary)
            wage_stats = self._calculate_count_mode_wages(current_state, new_permanent, new_temporary, converted_temps)
            nationality_stats = self._calculate_count_mode_nationalities(current_state, new_permanent, new_temporary, converted_temps)
            
            permanent_count = current_state.permanent_workers + new_permanent + converted_temps
            temporary_count = current_state.temporary_workers + new_temporary - converted_temps
        
        total_workers = permanent_count + temporary_count
        
        # Calculate queue backlogs by country (NEW FOR SPEC-5)
        queue_backlogs = self._calculate_queue_backlogs()
        
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
            total_wage_bill=wage_stats.total_wage_bill,
            top_temp_nationalities=nationality_stats.get_top_temporary_nationalities(),  # FROM SPEC-4
            converted_by_country=conversions_by_country,  # NEW FOR SPEC-5
            queue_backlog_by_country=queue_backlogs,  # NEW FOR SPEC-5
            country_cap_enabled=self.country_cap_enabled  # NEW FOR SPEC-5
        )
        
        self.states.append(next_state)
        
        logger.debug(f"Year {next_year}: Total={total_workers}, "
                    f"Permanent={permanent_count}, Temporary={temporary_count}, "
                    f"Converted={converted_temps}, Avg Wage=${wage_stats.avg_wage_total:,.0f}")
        
        return next_state
    
    def _calculate_queue_backlogs(self) -> Dict[str, int]:
        """
        Calculate current queue backlogs by country (NEW FOR SPEC-5).
        
        Returns:
            Dictionary mapping nationality to queue size
        """
        if self.country_cap_enabled:
            return {nationality: len(queue) for nationality, queue in self.country_queues.items()}
        elif self.global_queue is not None:
            # For uncapped mode, calculate backlogs by nationality from global queue
            backlogs = defaultdict(int)
            for temp_worker in self.global_queue:
                backlogs[temp_worker.nationality] += 1
            return dict(backlogs)
        else:
            return {}
    
    def _process_agent_mode_step(self, next_year: int, new_permanent: int, new_temporary: int) -> Tuple[int, Dict[str, int]]:
        """
        Process one simulation step in agent-mode.
        
        Args:
            next_year: The year being simulated
            new_permanent: Number of new permanent workers to add
            new_temporary: Number of new temporary workers to add
            
        Returns:
            Tuple of (total conversions, conversions by country)
        """
        # 1. Add new permanent workers (all U.S. nationals per SPEC-4)
        for _ in range(new_permanent):
            worker = Worker(
                id=self.next_worker_id,
                status=WorkerStatus.PERMANENT,
                nationality=PERMANENT_NATIONALITY,  # FROM SPEC-4
                age=self.rng.integers(25, 65),
                wage=STARTING_WAGE,
                created_year=next_year,
                entry_year=next_year,
                year_joined=next_year
            )
            self.workers.append(worker)
            self.next_worker_id += 1
        
        # 2. Add new temporary workers with nationality distribution (FROM SPEC-4)
        for _ in range(new_temporary):
            nationality = self._sample_nationality(is_temporary=True)
            worker = Worker(
                id=self.next_worker_id,
                status=WorkerStatus.TEMPORARY,
                nationality=nationality,  # FROM SPEC-4
                age=self.rng.integers(25, 55),
                wage=STARTING_WAGE,
                created_year=next_year,
                entry_year=next_year,
                year_joined=next_year
            )
            self.workers.append(worker)
            
            # Add to appropriate conversion queue (NEW FOR SPEC-5)
            self._add_to_conversion_queue(worker.id, nationality, next_year)
            self.next_worker_id += 1
        
        # 3. Process job changes and wage updates (FROM SPEC-3)
        self._process_job_changes()
        
        # 4. Process temporary-to-permanent conversions with per-country caps (NEW FOR SPEC-5)
        converted_temps, conversions_by_country = self._process_green_card_conversions()
        
        return converted_temps, conversions_by_country
    
    def _process_green_card_conversions(self) -> Tuple[int, Dict[str, int]]:
        """
        Process green card conversions with optional per-country caps (NEW FOR SPEC-5).
        
        Returns:
            Tuple of (total conversions, conversions by country)
        """
        conversions_by_country = defaultdict(int)
        converted_worker_ids = set()
        
        if self.country_cap_enabled:
            # Capped mode: process each nationality separately
            for nationality, queue in self.country_queues.items():
                country_conversions = 0
                country_limit = min(self.per_country_cap, len(queue))
                
                for _ in range(country_limit):
                    if queue:
                        temp_worker = queue.popleft()
                        converted_worker_ids.add(temp_worker.worker_id)
                        country_conversions += 1
                
                if country_conversions > 0:
                    conversions_by_country[nationality] = country_conversions
        else:
            # Uncapped mode: process global queue up to total cap
            total_conversions = min(self.annual_conversion_cap, len(self.global_queue))
            
            for _ in range(total_conversions):
                if self.global_queue:
                    temp_worker = self.global_queue.popleft()
                    converted_worker_ids.add(temp_worker.worker_id)
                    conversions_by_country[temp_worker.nationality] += 1
        
        # Update worker status for converted workers (nationality unchanged per SPEC-4)
        for worker in self.workers:
            if worker.id in converted_worker_ids:
                worker.convert_to_permanent()  # Uses method that preserves nationality
        
        total_converted = sum(conversions_by_country.values())
        return total_converted, dict(conversions_by_country)
    
    def _process_job_changes(self) -> None:
        """
        Process job changes and wage updates for all workers (FROM SPEC-3).
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
    
    def _process_count_mode_step(self, next_year: int, new_permanent: int, new_temporary: int) -> Tuple[int, Dict[str, int]]:
        """
        Process one simulation step in count-mode (approximation).
        
        Args:
            next_year: The year being simulated  
            new_permanent: Number of new permanent workers to add
            new_temporary: Number of new temporary workers to add
            
        Returns:
            Tuple of (total conversions, conversions by country)
        """
        # Add new temporary workers to queues (approximated)
        for i in range(new_temporary):
            # Sample nationality for approximation
            nationality = self._sample_nationality(is_temporary=True)
            self._add_to_conversion_queue(self.next_worker_id + i, nationality, next_year)
        
        self.next_worker_id += new_temporary
        
        # Process conversions with per-country caps
        return self._process_green_card_conversions_count_mode()
    
    def _process_green_card_conversions_count_mode(self) -> Tuple[int, Dict[str, int]]:
        """
        Process green card conversions in count-mode (NEW FOR SPEC-5).
        
        Returns:
            Tuple of (total conversions, conversions by country)
        """
        conversions_by_country = defaultdict(int)
        
        if self.country_cap_enabled:
            # Capped mode: process each nationality separately
            for nationality, queue in self.country_queues.items():
                country_limit = min(self.per_country_cap, len(queue))
                
                for _ in range(country_limit):
                    if queue:
                        queue.popleft()
                        conversions_by_country[nationality] += 1
        else:
            # Uncapped mode: process global queue up to total cap
            total_conversions = min(self.annual_conversion_cap, len(self.global_queue))
            
            # Approximate conversions by nationality based on queue composition
            nationality_counts = defaultdict(int)
            temp_queue = list(self.global_queue)
            
            for temp_worker in temp_queue:
                nationality_counts[temp_worker.nationality] += 1
            
            # Distribute conversions proportionally
            total_in_queue = len(self.global_queue)
            if total_in_queue > 0:
                for nationality, count in nationality_counts.items():
                    proportion = count / total_in_queue
                    country_conversions = round(total_conversions * proportion)
                    if country_conversions > 0:
                        conversions_by_country[nationality] = country_conversions
            
            # Remove converted workers from global queue
            for _ in range(total_conversions):
                if self.global_queue:
                    self.global_queue.popleft()
        
        total_converted = sum(conversions_by_country.values())
        return total_converted, dict(conversions_by_country)
    
    def _calculate_count_mode_wages(self, current_state: SimulationState, 
                                  new_permanent: int, new_temporary: int, 
                                  converted_temps: int) -> WageStatistics:
        """
        Calculate approximate wage statistics for count-mode (FROM SPEC-3).
        
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
    
    def _calculate_count_mode_nationalities(self, current_state: SimulationState,
                                          new_permanent: int, new_temporary: int,
                                          converted_temps: int) -> NationalityStatistics:
        """
        Calculate approximate nationality statistics for count-mode (FROM SPEC-4).
        
        Returns:
            NationalityStatistics with approximated values
        """
        # Approximate nationality distributions
        permanent_nationalities = {PERMANENT_NATIONALITY: current_state.permanent_workers + new_permanent + converted_temps}
        
        # Approximate temporary nationalities maintaining proportions
        temp_total = current_state.temporary_workers + new_temporary - converted_temps
        temporary_nationalities = {
            nationality: round(proportion * temp_total)
            for nationality, proportion in self.temp_nationality_distribution.items()
        }
        
        return NationalityStatistics(
            total_workers=current_state.total_workers + new_permanent + new_temporary,
            permanent_nationalities=permanent_nationalities,
            temporary_nationalities=temporary_nationalities
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
                
                # NEW FOR SPEC-5: Log per-country conversion information
                if self.country_cap_enabled and current_state.converted_by_country:
                    top_conversions = sorted(current_state.converted_by_country.items(), 
                                           key=lambda x: x[1], reverse=True)[:3]
                    logger.info(f"Top conversions by country: {top_conversions}")
        
        final_state = self.states[-1]
        total_conversions = sum(state.converted_temps for state in self.states[1:])
        logger.info(f"Simulation complete. Final workforce: {final_state.total_workers:,} "
                   f"({final_state.h1b_share:.2%} H-1B), "
                   f"{total_conversions} total conversions, "
                   f"Final avg wage: ${final_state.avg_wage_total:,.0f}")
        
        # NEW FOR SPEC-5: Log per-country cap summary
        if self.country_cap_enabled:
            total_country_conversions = sum(
                sum(state.converted_by_country.values()) 
                for state in self.states[1:]
            )
            final_backlogs = final_state.queue_backlog_by_country
            if final_backlogs:
                top_backlogs = sorted(final_backlogs.items(), key=lambda x: x[1], reverse=True)[:3]
                logger.info(f"Final queue backlogs by country (top 3): {top_backlogs}")
        
        # Print final nationality summary if requested (FROM SPEC-4)
        if self.config.show_nationality_summary and self.config.agent_mode:
            final_nationality_stats = NationalityStatistics.calculate(self.workers)
            self._print_nationality_summary("FINAL", final_nationality_stats)
        
        return self.states
    
    def _print_nationality_summary(self, label: str, nationality_stats: NationalityStatistics) -> None:
        """
        Print nationality breakdown summary (FROM SPEC-4).
        
        Args:
            label: Label for the summary (e.g., "INITIAL", "FINAL")
            nationality_stats: NationalityStatistics object
        """
        print(f"\n{label} NATIONALITY DISTRIBUTION")
        print("="*50)
        
        # Permanent worker nationalities
        print("Permanent worker nationalities:")
        total_perm = sum(nationality_stats.permanent_nationalities.values())
        if total_perm > 0:
            for nationality, count in sorted(nationality_stats.permanent_nationalities.items(), 
                                           key=lambda x: x[1], reverse=True):
                percentage = (count / total_perm) * 100
                print(f"  {nationality}: {percentage:.1f%} ({count:,} workers)")
        else:
            print("  No permanent workers")
        
        # Temporary worker nationalities
        print("\nTemporary worker nationalities:")
        temp_distribution = nationality_stats.get_temporary_distribution()
        total_temp = sum(nationality_stats.temporary_nationalities.values())
        if temp_distribution:
            for nationality, proportion in sorted(temp_distribution.items(), 
                                                key=lambda x: x[1], reverse=True):
                count = nationality_stats.temporary_nationalities[nationality]
                print(f"  {nationality}: {proportion:.1%} ({count:,} workers)")
        else:
            print("  No temporary workers")
        
        # NEW FOR SPEC-5: Print queue backlog information if per-country cap is enabled
        if self.country_cap_enabled:
            print("\nConversion queue backlogs:")
            queue_backlogs = self._calculate_queue_backlogs()
            if queue_backlogs:
                for nationality, backlog in sorted(queue_backlogs.items(), 
                                                 key=lambda x: x[1], reverse=True):
                    print(f"  {nationality}: {backlog:,} workers in queue")
            else:
                print("  No workers in conversion queues")
        
        print("="*50)
    
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
        Calculate annualized wage growth rate (FROM SPEC-3).
        
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
            # Check total conversion cap is never exceeded
            if state.converted_temps > self.annual_conversion_cap:
                logger.error(f"Year {state.year}: Total conversions {state.converted_temps} "
                           f"exceed cap {self.annual_conversion_cap}")
                return False
            
            # NEW FOR SPEC-5: Check per-country caps if enabled
            if state.country_cap_enabled:
                for nationality, conversions in state.converted_by_country.items():
                    if conversions > self.per_country_cap:
                        logger.error(f"Year {state.year}: Country {nationality} conversions {conversions} "
                                   f"exceed per-country cap {self.per_country_cap}")
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
        Validate wage calculations and consistency (FROM SPEC-3).
        
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
    
    def validate_nationality_consistency(self) -> bool:
        """
        Validate nationality distribution consistency (FROM SPEC-4).
        
        Returns:
            True if nationality distributions are consistent
        """
        if not self.config.agent_mode:
            return True  # Skip validation for count-mode
        
        for worker in self.workers:
            # Check that all workers have valid nationalities
            if not worker.nationality or not isinstance(worker.nationality, str):
                logger.error(f"Worker {worker.id} has invalid nationality: {worker.nationality}")
                return False
        
        # Check that temporary worker nationality distribution is reasonable
        temp_workers = [w for w in self.workers if w.is_temporary]
        if temp_workers:
            nationality_counts = {}
            for worker in temp_workers:
                nationality_counts[worker.nationality] = nationality_counts.get(worker.nationality, 0) + 1
            
            total_temp = len(temp_workers)
            for nationality, expected_prop in self.temp_nationality_distribution.items():
                actual_count = nationality_counts.get(nationality, 0)
                actual_prop = actual_count / total_temp
                
                # Allow some deviation due to randomness
                if abs(actual_prop - expected_prop) > 0.05:  # 5% tolerance
                    logger.warning(f"Nationality {nationality} proportion: expected {expected_prop:.2%}, "
                                 f"got {actual_prop:.2%}")
        
        return True
    
    def validate_country_cap_consistency(self) -> bool:
        """
        Validate per-country cap consistency (NEW FOR SPEC-5).
        
        Returns:
            True if per-country cap logic is consistent
        """
        if not self.country_cap_enabled:
            return True  # No validation needed for uncapped mode
        
        # Check that queue structure is consistent with cap mode
        if self.global_queue is not None:
            logger.error("Global queue should be None when country cap is enabled")
            return False
        
        # Check that conversions respect per-country limits
        for state in self.states[1:]:  # Skip initial state
            if state.country_cap_enabled:
                for nationality, conversions in state.converted_by_country.items():
                    if conversions > self.per_country_cap:
                        logger.error(f"Year {state.year}: {nationality} exceeded per-country cap "
                                   f"({conversions} > {self.per_country_cap})")
                        return False
        
        return True
    
    def to_agent_model(self) -> List[Worker]:
        """
        Return current worker agents (updated for SPEC-5).
        
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
        Updated for SPEC-5 to include per-country cap statistics.
        
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
        
        # Wage growth statistics (FROM SPEC-3)
        wage_growth = final_state.avg_wage_total - initial_state.avg_wage_total
        wage_growth_rate = self.get_wage_growth_rate(years_simulated) or 0
        
        # Nationality statistics (FROM SPEC-4)
        nationality_stats = {}
        if self.config.agent_mode:
            current_nationality_stats = NationalityStatistics.calculate(self.workers)
            nationality_stats = {
                "initial_temp_nationalities": initial_state.top_temp_nationalities,
                "final_temp_nationalities": final_state.top_temp_nationalities,
                "permanent_nationalities": dict(current_nationality_stats.permanent_nationalities),
                "temporary_nationalities": dict(current_nationality_stats.temporary_nationalities)
            }
        
        # Per-country cap statistics (NEW FOR SPEC-5)
        country_cap_stats = {}
        if self.country_cap_enabled:
            # Calculate total conversions by country across all years
            total_conversions_by_country = defaultdict(int)
            for state in self.states[1:]:
                for nationality, conversions in state.converted_by_country.items():
                    total_conversions_by_country[nationality] += conversions
            
            country_cap_stats = {
                "country_cap_enabled": True,
                "per_country_cap": self.per_country_cap,
                "per_country_cap_rate": PER_COUNTRY_CAP_SHARE,
                "total_conversions_by_country": dict(total_conversions_by_country),
                "final_queue_backlogs": final_state.queue_backlog_by_country,
                "countries_with_backlogs": len([c for c in final_state.queue_backlog_by_country.values() if c > 0])
            }
        else:
            country_cap_stats = {
                "country_cap_enabled": False,
                "per_country_cap": None,
                "per_country_cap_rate": None
            }
        
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
            # FROM SPEC-3: Wage statistics
            "initial_avg_wage": initial_state.avg_wage_total,
            "final_avg_wage": final_state.avg_wage_total,
            "total_wage_growth": wage_growth,
            "average_annual_wage_growth_rate": wage_growth_rate,
            "final_wage_bill": final_state.total_wage_bill,
            "job_change_prob_permanent": self.job_change_prob_perm,
            "job_change_prob_temporary": self.job_change_prob_temp,
            # FROM SPEC-4: Nationality statistics
            **nationality_stats,
            # NEW FOR SPEC-5: Per-country cap statistics
            **country_cap_stats
        }
