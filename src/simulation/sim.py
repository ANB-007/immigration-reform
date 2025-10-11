# src/simulation/sim.py
"""
Core simulation engine for workforce growth modeling.
Handles worker dynamics, green card conversions, and wage tracking.
SPEC-10: Fixed to use empirical parameters and streamlined models.
"""

import logging
import math
from typing import List, Optional, Dict, Tuple
from collections import defaultdict, deque
import numpy as np

from .models import (
    SimulationConfig, SimulationState, Worker, WorkerStatus, 
    TemporaryWorker, WageStatistics, NationalityStatistics
)
from .empirical_params import (
    H1B_SHARE, ANNUAL_PERMANENT_ENTRY_RATE, ANNUAL_H1B_ENTRY_RATE,
    GREEN_CARD_CAP_ABS, REAL_US_WORKFORCE_SIZE, PER_COUNTRY_CAP_SHARE,
    STARTING_WAGE, JOB_CHANGE_PROB_PERM, JOB_CHANGE_PROB_TEMP,
    WAGE_JUMP_FACTOR_MEAN_PERM, WAGE_JUMP_FACTOR_STD_PERM,
    WAGE_JUMP_FACTOR_MEAN_TEMP, WAGE_JUMP_FACTOR_STD_TEMP,
    TEMP_NATIONALITY_DISTRIBUTION, PERMANENT_NATIONALITY,
    CARRYOVER_FRACTION_STRATEGY, calculate_annual_conversion_cap,
    calculate_per_country_caps_deterministic, DEFAULT_SIMULATION_START_YEAR,
    CONVERSION_WAGE_BUMP
)

logger = logging.getLogger(__name__)

class Simulation:
    """
    Core simulation engine for workforce growth modeling.
    SPEC-10: Streamlined with empirical parameters and fixed data models.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize simulation with configuration.
        SPEC-10: Fixed to use empirical parameter functions and proper Worker initialization.
        
        Args:
            config: SimulationConfig object with simulation parameters
        """
        self.config = config
        self.current_year = config.start_year
        self.states: List[SimulationState] = []
        self.workers: List[Worker] = []
        self.next_worker_id = 1
        
        # SPEC-10: Single RNG for all stochastic operations
        self.rng = np.random.default_rng(config.seed)
        
        # Load nationality distribution
        self.temp_nationality_distribution = TEMP_NATIONALITY_DISTRIBUTION.copy()
        
        # SPEC-10: Fixed annual conversion cap using empirical function
        self.annual_sim_cap = calculate_annual_conversion_cap(config.initial_workers)
        
        logger.info(f"Annual conversion cap: {self.annual_sim_cap}")
        
        # SPEC-10: Per-country cap system
        self.country_cap_enabled = config.country_cap_enabled
        if self.country_cap_enabled:
            nationalities = list(self.temp_nationality_distribution.keys())
            self.per_country_caps = calculate_per_country_caps_deterministic(
                self.annual_sim_cap, nationalities
            )
            
            # Verify total
            total_per_country_slots = sum(self.per_country_caps.values())
            assert total_per_country_slots == self.annual_sim_cap, \
                f"Per-country caps sum to {total_per_country_slots}, expected {self.annual_sim_cap}"
            
            logger.info(f"Per-country caps: {self.per_country_caps}")
        else:
            self.per_country_caps = {}
        
        # SPEC-10: Synchronized queue management
        self.global_queue = deque()  # FIFO order for uncapped mode
        self.country_queues = {
            nationality: deque() for nationality in self.temp_nationality_distribution.keys()
        }  # Per-country queues for capped mode
        
        if self.country_cap_enabled:
            logger.info("Per-country cap ENABLED: using nationality-specific FIFO queues")
        else:
            logger.info("Per-country cap DISABLED: using global FIFO queue")
        
        # Cumulative conversion tracking
        self.cumulative_conversions = 0
        
        # Initialize workforce
        self._initialize_workforce()
        
        # Validation
        self._validate_nationality_distribution()
        
        logger.info(f"Initialized simulation with {config.initial_workers:,} workers")
        logger.info(f"Initial H-1B share: {self.states[0].h1b_share:.3%}")
    
    def _initialize_workforce(self) -> None:
        """
        Initialize the workforce with proper H-1B/permanent split.
        SPEC-10: Fixed Worker initialization with proper parameters.
        """
        initial_temporary = round(self.config.initial_workers * H1B_SHARE)
        initial_permanent = self.config.initial_workers - initial_temporary
        
        # Handle edge case for very small initial populations
        if self.config.initial_workers < 100:
            logger.warning(
                f"Small initial population ({self.config.initial_workers}). "
                "Results may show discretization effects."
            )
        
        # SPEC-10: Create deterministic temporary worker list ONCE
        temp_workers_list = []
        
        # Agent-mode: Create individual Worker objects
        if self.config.agent_mode:
            # SPEC-10: Fixed Worker initialization with proper parameters
            # Create permanent workers (all U.S. nationals)
            for i in range(initial_permanent):
                worker = Worker(
                    id=self.next_worker_id,
                    status=WorkerStatus.PERMANENT,
                    nationality=str(PERMANENT_NATIONALITY),
                    simulation_start_year=self.current_year,  # SPEC-10: Required parameter
                    entry_year_offset=0,                      # SPEC-10: Required parameter
                    age=self.rng.integers(25, 65),
                    wage=STARTING_WAGE
                )
                self.workers.append(worker)
                self.next_worker_id += 1
            
            # Create temporary workers with deterministic nationality assignment
            for i in range(initial_temporary):
                nationality = self._sample_nationality(is_temporary=True)
                worker = Worker(
                    id=self.next_worker_id,
                    status=WorkerStatus.TEMPORARY,
                    nationality=str(nationality),
                    simulation_start_year=self.current_year,  # SPEC-10: Required parameter
                    entry_year_offset=0,                      # SPEC-10: Required parameter
                    age=self.rng.integers(25, 55),
                    wage=STARTING_WAGE
                )
                self.workers.append(worker)
                
                # Add to deterministic list
                temp_worker = TemporaryWorker(worker.id, self.current_year, nationality)
                temp_workers_list.append(temp_worker)
                self.next_worker_id += 1
            
            # SPEC-10: Calculate initial statistics using WageStatistics.calculate
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
                permanent_nationalities={str(PERMANENT_NATIONALITY): initial_permanent},
                temporary_nationalities={k: round(v * initial_temporary) for k, v in self.temp_nationality_distribution.items()}
            )
            
            # Create deterministic temp worker list for count mode
            for nationality, proportion in self.temp_nationality_distribution.items():
                temp_workers_for_nationality = round(proportion * initial_temporary)
                for i in range(temp_workers_for_nationality):
                    temp_worker = TemporaryWorker(self.next_worker_id, self.current_year, nationality)
                    temp_workers_list.append(temp_worker)
                    self.next_worker_id += 1
        
        # SPEC-10: Add the SAME workers to both queues in the SAME order
        for temp_worker in temp_workers_list:
            self.global_queue.append(temp_worker)
            self.country_queues[temp_worker.nationality].append(temp_worker)
        
        # SPEC-10: Create initial state using new SimulationState with workers list
        initial_state = SimulationState(
            year=self.current_year,
            workers=self.workers.copy(),
            annual_conversion_cap=self.annual_sim_cap,
            new_permanent=0,
            new_temporary=0,
            converted_temps=0,
            cumulative_conversions=0,
            country_cap_enabled=self.country_cap_enabled
        )
        
        self.states.append(initial_state)
        
        # Print nationality summary if requested
        if self.config.show_nationality_summary:
            self._print_nationality_summary("INITIAL", nationality_stats)
    
    def _sample_nationality(self, is_temporary: bool) -> str:
        """Sample a nationality for a new worker."""
        if not is_temporary:
            return str(PERMANENT_NATIONALITY)
        
        nationalities = list(self.temp_nationality_distribution.keys())
        probabilities = list(self.temp_nationality_distribution.values())
        
        selected = self.rng.choice(nationalities, p=probabilities)
        return str(selected)
    
    def step(self) -> SimulationState:
        """
        Execute one simulation step (one year).
        SPEC-10: Fixed to use proper SimulationState properties.
        
        Returns:
            SimulationState representing the state after this step
        """
        next_year = self.current_year + 1
        
        # Calculate new entries based on current total workforce
        current_total = self.states[-1].total_workers
        new_permanent = round(current_total * ANNUAL_PERMANENT_ENTRY_RATE)
        new_temporary = round(current_total * ANNUAL_H1B_ENTRY_RATE)
        
        if hasattr(self.config, 'debug') and self.config.debug:
            logger.info(f"Year {next_year}: Adding {new_permanent} permanent, {new_temporary} temporary workers")
        
        if self.config.agent_mode:
            converted_temps, conversions_by_country = self._process_agent_mode_step(
                next_year, new_permanent, new_temporary
            )
        else:
            converted_temps, conversions_by_country = self._process_count_mode_step(
                next_year, new_permanent, new_temporary
            )
        
        # Update cumulative conversions
        self.cumulative_conversions += converted_temps
        
        # SPEC-10: Create new state with updated workers list
        new_state = SimulationState(
            year=next_year,
            workers=self.workers.copy(),
            annual_conversion_cap=self.annual_sim_cap,
            new_permanent=new_permanent,
            new_temporary=new_temporary,
            converted_temps=converted_temps,
            cumulative_conversions=self.cumulative_conversions,
            country_cap_enabled=self.country_cap_enabled
        )
        
        # SPEC-10: Debug output using properties
        if hasattr(self.config, 'debug') and self.config.debug:
            total_backlog_uncapped = len(self.global_queue)
            total_backlog_capped = sum(len(queue) for queue in self.country_queues.values())
            logger.info(f"Year {next_year}: slots={self.annual_sim_cap}, converted={converted_temps}, "
                       f"queue_total_len_uncapped={total_backlog_uncapped}, "
                       f"backlog_total_len_capped={total_backlog_capped}")
            if self.country_cap_enabled:
                logger.info(f"Per-country conversions: {conversions_by_country}, sum={sum(conversions_by_country.values())}")
        
        self.states.append(new_state)
        self.current_year = next_year
        
        return new_state
    
    def _process_agent_mode_step(self, next_year: int, new_permanent: int, new_temporary: int) -> Tuple[int, Dict[str, int]]:
        """
        Process one simulation step in agent-mode.
        SPEC-10: Fixed Worker creation with proper parameters.
        """
        # 1. Add new permanent workers (all U.S. nationals)
        for _ in range(new_permanent):
            worker = Worker(
                id=self.next_worker_id,
                status=WorkerStatus.PERMANENT,
                nationality=str(PERMANENT_NATIONALITY),
                simulation_start_year=self.config.start_year,  # SPEC-10: Required parameter
                entry_year_offset=next_year - self.config.start_year,  # SPEC-10: Required parameter  
                age=self.rng.integers(25, 65),
                wage=STARTING_WAGE
            )
            self.workers.append(worker)
            self.next_worker_id += 1
        
        # 2. Add new temporary workers with nationality distribution
        new_temp_workers_list = []
        for _ in range(new_temporary):
            nationality = self._sample_nationality(is_temporary=True)
            worker = Worker(
                id=self.next_worker_id,
                status=WorkerStatus.TEMPORARY,
                nationality=str(nationality),
                simulation_start_year=self.config.start_year,  # SPEC-10: Required parameter
                entry_year_offset=next_year - self.config.start_year,  # SPEC-10: Required parameter
                age=self.rng.integers(25, 55),
                wage=STARTING_WAGE
            )
            self.workers.append(worker)
            
            temp_worker = TemporaryWorker(worker.id, next_year, nationality)
            new_temp_workers_list.append(temp_worker)
            self.next_worker_id += 1
        
        # Add new workers to both queues from the SAME list
        for temp_worker in new_temp_workers_list:
            self.global_queue.append(temp_worker)
            self.country_queues[temp_worker.nationality].append(temp_worker)
        
        # 3. Process job changes and wage updates
        self._process_job_changes(next_year)
        
        # 4. Process conversions with fixed caps
        converted_temps, conversions_by_country = self._process_green_card_conversions(next_year)
        
        return converted_temps, conversions_by_country
    
    def _process_job_changes(self, current_year: int) -> None:
        """
        Process job changes and wage updates for all workers.
        SPEC-10: Fixed with proper conversion wage bump application.
        """
        for worker in self.workers:
            # Determine job change probability and wage parameters based on current status
            if worker.is_permanent:
                # Check if this is a converted worker and timing for parameter switch
                if worker.was_converted and worker.conversion_year is not None:
                    if current_year > worker.conversion_year:
                        # Use permanent parameters (year after conversion)
                        job_change_prob = JOB_CHANGE_PROB_PERM
                        wage_mean = WAGE_JUMP_FACTOR_MEAN_PERM
                        wage_std = WAGE_JUMP_FACTOR_STD_PERM
                    else:
                        # Still use temporary parameters (conversion year)
                        job_change_prob = JOB_CHANGE_PROB_TEMP
                        wage_mean = WAGE_JUMP_FACTOR_MEAN_TEMP
                        wage_std = WAGE_JUMP_FACTOR_STD_TEMP
                else:
                    # Initially permanent worker
                    job_change_prob = JOB_CHANGE_PROB_PERM
                    wage_mean = WAGE_JUMP_FACTOR_MEAN_PERM
                    wage_std = WAGE_JUMP_FACTOR_STD_PERM
            else:
                # Temporary worker
                job_change_prob = JOB_CHANGE_PROB_TEMP
                wage_mean = WAGE_JUMP_FACTOR_MEAN_TEMP
                wage_std = WAGE_JUMP_FACTOR_STD_TEMP
            
            # Check if worker changes jobs
            if self.rng.random() < job_change_prob:
                jump_factor = max(1.0, self.rng.normal(wage_mean, wage_std))
                worker.apply_wage_jump(jump_factor)
    
    def _process_green_card_conversions(self, current_year: int) -> Tuple[int, Dict[str, int]]:
        """
        Process temporary-to-permanent conversions with fixed caps.
        SPEC-10: Fixed conversion logic with proper wage bump.
        """
        slots_this_year = self.annual_sim_cap
        
        conversions_by_country = {}
        total_conversions = 0
        
        # Track which workers are converted for perfect synchronization
        converted_worker_ids = set()
        
        if self.country_cap_enabled:
            # Per-country capped conversions
            for nationality in self.temp_nationality_distribution.keys():
                if nationality not in self.country_queues:
                    continue
                
                queue = self.country_queues[nationality]
                per_country_cap = self.per_country_caps.get(nationality, 0)
                
                # Convert up to per-country cap from this nationality
                conversions_this_country = min(len(queue), per_country_cap)
                
                # Convert workers from this country
                for _ in range(conversions_this_country):
                    if queue:
                        temp_worker = queue.popleft()
                        converted_worker_ids.add(temp_worker.worker_id)
                        
                        # Find and convert the actual worker
                        worker = next((w for w in self.workers if w.id == temp_worker.worker_id), None)
                        if worker and worker.is_temporary:
                            worker.convert_to_permanent(current_year)
                            # SPEC-10: Apply conversion wage bump
                            worker.apply_wage_jump(CONVERSION_WAGE_BUMP)
                
                conversions_by_country[nationality] = conversions_this_country
                total_conversions += conversions_this_country
        
        else:
            # Global uncapped conversions (with fixed annual cap)
            for _ in range(slots_this_year):
                if not self.global_queue:
                    break
                
                temp_worker = self.global_queue.popleft()
                converted_worker_ids.add(temp_worker.worker_id)
                
                # Find and convert the actual worker
                worker = next((w for w in self.workers if w.id == temp_worker.worker_id), None)
                if worker and worker.is_temporary:
                    worker.convert_to_permanent(current_year)
                    # SPEC-10: Apply conversion wage bump
                    worker.apply_wage_jump(CONVERSION_WAGE_BUMP)
                    nationality = worker.nationality
                    conversions_by_country[nationality] = conversions_by_country.get(nationality, 0) + 1
                    total_conversions += 1
        
        # Remove the EXACT SAME workers from both queue structures
        self.global_queue = deque([tw for tw in self.global_queue if tw.worker_id not in converted_worker_ids])
        
        for nationality in self.country_queues:
            self.country_queues[nationality] = deque([tw for tw in self.country_queues[nationality] 
                                                     if tw.worker_id not in converted_worker_ids])
        
        return total_conversions, conversions_by_country
    
    def _process_count_mode_step(self, next_year: int, new_permanent: int, new_temporary: int) -> Tuple[int, Dict[str, int]]:
        """
        Process one simulation step in count-mode.
        SPEC-10: Fixed with synchronized queues.
        """
        # Add new temporary workers to both queues from the SAME list
        new_temp_workers_list = []
        for _ in range(new_temporary):
            nationality = self._sample_nationality(is_temporary=True)
            temp_worker = TemporaryWorker(self.next_worker_id, next_year, nationality)
            new_temp_workers_list.append(temp_worker)
            self.next_worker_id += 1
        
        # Add to queues from the same list
        for temp_worker in new_temp_workers_list:
            self.global_queue.append(temp_worker)
            self.country_queues[temp_worker.nationality].append(temp_worker)
        
        # Process conversions using same logic as agent mode
        return self._process_green_card_conversions(next_year)
    
    def _calculate_queue_backlogs(self) -> Dict[str, int]:
        """
        Calculate current queue backlogs by nationality.
        SPEC-10: Proper backlog calculation for both modes.
        
        Returns:
            Dictionary mapping nationality to queue size
        """
        backlogs = {}
        
        if self.country_cap_enabled:
            # Use country-specific queues
            for nationality, queue in self.country_queues.items():
                backlogs[nationality] = len(queue)
        else:
            # For uncapped mode, calculate backlog by nationality from global queue
            nationality_counts = defaultdict(int)
            for temp_worker in self.global_queue:
                nationality_counts[temp_worker.nationality] += 1
            
            # Ensure all nationalities are represented
            for nationality in self.temp_nationality_distribution.keys():
                backlogs[nationality] = nationality_counts[nationality]
        
        return backlogs
    
    def _validate_nationality_distribution(self) -> None:
        """Validate that nationality distribution sums to 1.0."""
        total = sum(self.temp_nationality_distribution.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Nationality distribution must sum to 1.0, got {total}")
        
        logger.info(f"Nationality distribution validated. "
                   f"Top nationalities: {dict(list(self.temp_nationality_distribution.items())[:3])}")
    
    def _print_nationality_summary(self, label: str, nationality_stats: NationalityStatistics) -> None:
        """Print nationality breakdown summary."""
        print(f"\n{label} NATIONALITY DISTRIBUTION")
        print("="*50)
        
        # Permanent worker nationalities
        print("Permanent worker nationalities:")
        total_perm = sum(nationality_stats.permanent_nationalities.values())
        if total_perm > 0:
            for nationality, count in sorted(nationality_stats.permanent_nationalities.items(), 
                                           key=lambda x: x[1], reverse=True):
                percentage = (count / total_perm) * 100
                print(f"  {nationality}: {percentage:.1f}% ({count:,} workers)")
        else:
            print("  No permanent workers")
        
        # Temporary worker nationalities  
        print("\nTemporary worker nationalities:")
        # SPEC-10: Fixed to use get_top_temporary_nationalities instead of get_temporary_distribution
        temp_distribution = nationality_stats.get_top_temporary_nationalities(10)
        if temp_distribution:
            for nationality, proportion in sorted(temp_distribution.items(), 
                                                key=lambda x: x[1], reverse=True):
                count = nationality_stats.temporary_nationalities.get(nationality, 0)
                percentage = proportion * 100
                print(f"  {nationality}: {percentage:.1f}% ({count:,} workers)")
        else:
            print("  No temporary workers")
        
        # Queue backlog information if per-country cap is enabled
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
    
    def run(self) -> List[SimulationState]:
        """Run the complete simulation."""
        if hasattr(self.config, 'debug') and self.config.debug:
            logger.info(f"SPEC-10 Debug Info:")
            logger.info(f"annual_sim_cap: {self.annual_sim_cap}")
            if self.country_cap_enabled:
                logger.info(f"per_country_caps: {self.per_country_caps}")
        
        logger.info(f"Starting {self.config.years}-year simulation...")
        
        for year in range(self.config.years):
            self.step()
            
            if (year + 1) % 10 == 0 or year == 0:
                state = self.states[-1]
                logger.info(f"Year {state.year}: {state.total_workers:,} workers, "
                           f"{state.converted_temps} conversions, "
                           f"${state.avg_wage_total:,.0f} avg wage")
        
        # Print final nationality summary if requested
        if self.config.show_nationality_summary and self.config.agent_mode:
            final_nationality_stats = NationalityStatistics.calculate(self.workers)
            self._print_nationality_summary("FINAL", final_nationality_stats)
        
        logger.info(f"Simulation completed. Final workforce: {self.states[-1].total_workers:,}")
        logger.info(f"Total conversions: {self.cumulative_conversions:,}")
        
        # Final debug output
        if hasattr(self.config, 'debug') and self.config.debug:
            total_backlog_uncapped = len(self.global_queue)
            total_backlog_capped = sum(len(queue) for queue in self.country_queues.values())
            logger.info(f"Final queue lengths - Uncapped: {total_backlog_uncapped}, Capped: {total_backlog_capped}")
            
            expected_total_conversions = self.annual_sim_cap * self.config.years
            logger.info(f"Expected total conversions: {expected_total_conversions}, Actual: {self.cumulative_conversions}")
        
        return self.states
    
    def to_agent_model(self) -> List[Worker]:
        """Convert simulation to agent-based model for detailed analysis."""
        if not self.config.agent_mode:
            logger.warning("Simulation not run in agent-mode. Worker data may be incomplete.")
        
        return self.workers.copy()
    
    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics for the simulation."""
        if not self.states:
            return {}
        
        initial_state = self.states[0]
        final_state = self.states[-1]
        
        # Calculate growth statistics
        total_growth = final_state.total_workers - initial_state.total_workers
        years_simulated = len(self.states) - 1
        
        if years_simulated > 0:
            avg_annual_growth_rate = (total_growth / initial_state.total_workers) / years_simulated
        else:
            avg_annual_growth_rate = 0.0
        
        # Calculate wage statistics
        wage_growth = final_state.avg_wage_total - initial_state.avg_wage_total
        if years_simulated > 0 and initial_state.avg_wage_total > 0:
            avg_annual_wage_growth_rate = ((final_state.avg_wage_total / initial_state.avg_wage_total) ** (1/years_simulated)) - 1
        else:
            avg_annual_wage_growth_rate = 0.0
        
        # Calculate conversion statistics
        total_conversions = sum(state.converted_temps for state in self.states[1:])
        max_possible_conversions = self.annual_sim_cap * years_simulated
        conversion_utilization = total_conversions / max_possible_conversions if max_possible_conversions > 0 else 0.0
        
        stats = {
            'years_simulated': years_simulated,
            'simulation_mode': 'Agent-based' if self.config.agent_mode else 'Count-based',
            'industry': 'Information Technology',
            'initial_workforce': initial_state.total_workers,
            'final_workforce': final_state.total_workers,
            'total_growth': total_growth,
            'average_annual_growth_rate': avg_annual_growth_rate,
            'initial_h1b_share': initial_state.h1b_share,
            'final_h1b_share': final_state.h1b_share,
            'h1b_share_change': final_state.h1b_share - initial_state.h1b_share,
            'total_new_permanent': sum(state.new_permanent for state in self.states[1:]),
            'total_new_temporary': sum(state.new_temporary for state in self.states[1:]),
            'total_conversions': total_conversions,
            'annual_conversion_cap': self.annual_sim_cap,
            'conversion_utilization': conversion_utilization,
            'initial_avg_wage': initial_state.avg_wage_total,
            'final_avg_wage': final_state.avg_wage_total,
            'total_wage_growth': wage_growth,
            'average_annual_wage_growth_rate': avg_annual_wage_growth_rate,
            'final_wage_bill': final_state.total_wage_bill,
            'job_change_prob_permanent': JOB_CHANGE_PROB_PERM,
            'job_change_prob_temporary': JOB_CHANGE_PROB_TEMP,
            'country_cap_enabled': self.country_cap_enabled
        }
        
        # Add per-country cap statistics if enabled
        if self.country_cap_enabled:
            stats.update({
                'per_country_caps': self.per_country_caps,
                'per_country_cap_rate': PER_COUNTRY_CAP_SHARE,
                'final_queue_backlogs': self._calculate_queue_backlogs(),
                'countries_with_backlogs': len([b for b in self._calculate_queue_backlogs().values() if b > 0])
            })
            
            # Calculate total conversions by country
            total_conversions_by_country = defaultdict(int)
            for state in self.states[1:]:
                # SPEC-10: Handle cases where converted_by_country might not exist
                if hasattr(state, 'converted_by_country'):
                    for nationality, conversions in state.converted_by_country.items():
                        total_conversions_by_country[nationality] += conversions
            
            stats['total_conversions_by_country'] = dict(total_conversions_by_country)
        
        return stats
    
    # Validation methods for testing
    def validate_fixed_conversion_invariants(self) -> bool:
        """
        Validate SPEC-10 invariants.
        
        Returns:
            True if all invariants hold
        """
        if len(self.states) < 2:
            return True
        
        # Invariant A: annual_sim_cap is constant every year
        for state in self.states:
            if state.annual_conversion_cap != self.annual_sim_cap:
                logger.error(f"Invariant A failed: Year {state.year} cap {state.annual_conversion_cap} != expected {self.annual_sim_cap}")
                return False
        
        # Check conversions don't exceed cap
        for state in self.states[1:]:
            if state.converted_temps > self.annual_sim_cap:
                logger.error(f"Invariant A failed: Year {state.year} conversions {state.converted_temps} > cap {self.annual_sim_cap}")
                return False
        
        return True
    
    def get_total_backlog_uncapped(self) -> int:
        """Get total backlog for uncapped mode (from global queue)."""
        return len(self.global_queue)
    
    def get_total_backlog_capped(self) -> int:
        """Get total backlog for capped mode (sum of country queues)."""
        return sum(len(queue) for queue in self.country_queues.values())
