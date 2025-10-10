# src/simulation/models.py
"""
Data models for the workforce simulation.
Defines Worker agents and related data structures.
Updated for SPEC-7 to include backlog analysis functionality.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Deque
from enum import Enum
from collections import deque

from .empirical_params import STARTING_WAGE, PERMANENT_NATIONALITY

class WorkerStatus(Enum):
    """Enumeration of worker status types."""
    PERMANENT = "permanent"
    TEMPORARY = "temporary"  # H-1B holders

@dataclass
class Worker:
    """
    Represents a single worker agent in the simulation.
    Updated for SPEC-7 with backlog analysis considerations.
    
    Attributes:
        id: Unique identifier for the worker
        status: Worker status (permanent or temporary/H-1B)
        nationality: Worker's nationality (immutable) (FROM SPEC-4)
        age: Age of the worker in years
        year_joined: Year the worker joined as temporary (for FIFO conversion)
        wage: Current wage in USD per year (FROM SPEC-3)
        created_year: Year the worker was created/hired (FROM SPEC-3)
        entry_year: Year the worker entered the workforce
        skills: Skill level or category (for future extensions)
        occupation: Job category or title (for future extensions)
        employer_id: ID of employing organization (for future extensions)
        attributes: Additional extensible attributes
    """
    id: int
    status: WorkerStatus
    nationality: str  # FROM SPEC-4: Immutable nationality
    age: int = 35  # Default age
    year_joined: int = 2025  # Year joined as temporary worker
    wage: float = STARTING_WAGE  # FROM SPEC-3: Current wage in USD per year
    created_year: int = 2025  # FROM SPEC-3: Year worker was created/hired
    entry_year: int = 2025  # Year entered workforce
    skills: Optional[List[str]] = None
    occupation: Optional[str] = None
    employer_id: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.age < 16 or self.age > 100:
            raise ValueError(f"Invalid age: {self.age}. Must be between 16 and 100.")
        
        if self.wage < 0:
            raise ValueError(f"Invalid wage: {self.wage}. Must be non-negative.")
        
        if not self.nationality or not isinstance(self.nationality, str):
            raise ValueError(f"Invalid nationality: {self.nationality}. Must be a non-empty string.")
        
        # Set year_joined to entry_year if not specified and worker is temporary
        if self.status == WorkerStatus.TEMPORARY and self.year_joined == 2025 and self.entry_year != 2025:
            self.year_joined = self.entry_year
            
        # Set created_year to entry_year if not specified
        if self.created_year == 2025 and self.entry_year != 2025:
            self.created_year = self.entry_year
    
    @property
    def is_permanent(self) -> bool:
        """Returns True if worker has permanent status."""
        return self.status == WorkerStatus.PERMANENT
    
    @property
    def is_temporary(self) -> bool:
        """Returns True if worker has temporary/H-1B status."""
        return self.status == WorkerStatus.TEMPORARY
    
    @property
    def is_us_national(self) -> bool:
        """Returns True if worker is a U.S. national (FROM SPEC-4)."""
        return self.nationality == PERMANENT_NATIONALITY
    
    def years_in_workforce(self, current_year: int) -> int:
        """Calculate years the worker has been in the workforce."""
        return max(0, current_year - self.entry_year)
    
    def years_as_temporary(self, current_year: int) -> int:
        """Calculate years the worker has been in temporary status."""
        if self.status != WorkerStatus.TEMPORARY:
            return 0
        return max(0, current_year - self.year_joined)
    
    def apply_wage_jump(self, jump_factor: float) -> None:
        """
        Apply wage jump due to job change (FROM SPEC-3).
        
        Args:
            jump_factor: Multiplicative factor for wage increase (e.g., 1.08 for 8% increase)
        """
        if jump_factor < 0:
            raise ValueError(f"Jump factor must be non-negative, got {jump_factor}")
        
        # Ensure wage doesn't decrease (minimum factor of 1.0)
        effective_factor = max(jump_factor, 1.0)
        self.wage *= effective_factor
    
    def convert_to_permanent(self) -> None:
        """
        Convert worker from temporary to permanent status (FROM SPEC-4).
        Nationality remains unchanged as per SPEC-4 requirements.
        """
        if self.status == WorkerStatus.TEMPORARY:
            self.status = WorkerStatus.PERMANENT
        else:
            raise ValueError(f"Cannot convert worker {self.id}: already permanent")

@dataclass
class SimulationState:
    """
    Represents the state of the simulation at a given time step.
    Updated for SPEC-7 to include backlog analysis statistics.
    
    Attributes:
        year: Current simulation year
        total_workers: Total number of workers
        permanent_workers: Number of permanent workers
        temporary_workers: Number of temporary/H-1B workers
        new_permanent: New permanent workers added this year
        new_temporary: New temporary workers added this year
        converted_temps: Temporary workers converted to permanent
        avg_wage_total: Average wage across all workers (FROM SPEC-3)
        avg_wage_permanent: Average wage for permanent workers (FROM SPEC-3)
        avg_wage_temporary: Average wage for temporary workers (FROM SPEC-3)
        total_wage_bill: Total wages paid to all workers (FROM SPEC-3)
        top_temp_nationalities: Top 3 nationalities for temporary workers (FROM SPEC-4)
        converted_by_country: Conversions by nationality this year (FROM SPEC-5)
        queue_backlog_by_country: Queue backlogs by nationality (FROM SPEC-5)
        country_cap_enabled: Whether per-country cap is active (FROM SPEC-5)
    """
    year: int
    total_workers: int
    permanent_workers: int
    temporary_workers: int
    new_permanent: int = 0
    new_temporary: int = 0
    converted_temps: int = 0
    avg_wage_total: float = 0.0  # FROM SPEC-3
    avg_wage_permanent: float = 0.0  # FROM SPEC-3
    avg_wage_temporary: float = 0.0  # FROM SPEC-3
    total_wage_bill: float = 0.0  # FROM SPEC-3
    top_temp_nationalities: Dict[str, float] = field(default_factory=dict)  # FROM SPEC-4
    converted_by_country: Dict[str, int] = field(default_factory=dict)  # FROM SPEC-5
    queue_backlog_by_country: Dict[str, int] = field(default_factory=dict)  # FROM SPEC-5
    country_cap_enabled: bool = False  # FROM SPEC-5
    
    def __post_init__(self):
        """Validation of state consistency."""
        if self.permanent_workers + self.temporary_workers != self.total_workers:
            raise ValueError("Sum of permanent and temporary workers must equal total workers")
        
        if any(count < 0 for count in [self.total_workers, self.permanent_workers, 
                                     self.temporary_workers, self.new_permanent, 
                                     self.new_temporary, self.converted_temps]):
            raise ValueError("All worker counts must be non-negative")
        
        # Validate wage statistics
        if any(wage < 0 for wage in [self.avg_wage_total, self.avg_wage_permanent, 
                                   self.avg_wage_temporary, self.total_wage_bill]):
            raise ValueError("All wage statistics must be non-negative")
        
        # Validate per-country statistics (FROM SPEC-5)
        if any(count < 0 for count in self.converted_by_country.values()):
            raise ValueError("All conversion counts must be non-negative")
        
        if any(count < 0 for count in self.queue_backlog_by_country.values()):
            raise ValueError("All queue backlog counts must be non-negative")

    @property 
    def h1b_share(self) -> float:
        """Calculate current H-1B share of workforce."""
        if self.total_workers == 0:
            return 0.0
        return self.temporary_workers / self.total_workers
    
    @property
    def permanent_share(self) -> float:
        """Calculate current permanent worker share of workforce."""
        if self.total_workers == 0:
            return 0.0
        return self.permanent_workers / self.total_workers

@dataclass 
class SimulationConfig:
    """Configuration parameters for a simulation run."""
    initial_workers: int
    years: int
    seed: Optional[int] = None
    live_fetch: bool = False
    output_path: str = "data/sample_output.csv"
    agent_mode: bool = True  # FROM SPEC-3: Default to agent-mode for wage tracking
    show_nationality_summary: bool = False  # FROM SPEC-4: Show nationality breakdown
    country_cap_enabled: bool = False  # FROM SPEC-5: Enable per-country cap
    compare_backlogs: bool = False  # NEW FOR SPEC-7: Enable backlog comparison analysis
    
    def __post_init__(self):
        """Validation of configuration parameters."""
        if self.initial_workers <= 0:
            raise ValueError("Initial workers must be positive")
        if self.years <= 0:
            raise ValueError("Simulation years must be positive")

@dataclass
class TemporaryWorker:
    """
    Represents a temporary worker in the conversion queue (FROM SPEC-2).
    Used for FIFO ordering of temporary-to-permanent conversions.
    Updated for SPEC-5 to support per-country queuing.
    """
    worker_id: int
    year_joined: int
    nationality: str = ""  # FROM SPEC-5: Added for per-country queue management
    
    def __lt__(self, other):
        """Less than comparison for sorting (earlier year_joined comes first)."""
        if self.year_joined != other.year_joined:
            return self.year_joined < other.year_joined
        # Use worker_id as tiebreaker for same year
        return self.worker_id < other.worker_id

@dataclass
class WageStatistics:
    """
    Container for wage statistics calculation (FROM SPEC-3).
    """
    total_workers: int
    permanent_workers: int
    temporary_workers: int
    avg_wage_total: float
    avg_wage_permanent: float
    avg_wage_temporary: float
    total_wage_bill: float
    
    @classmethod
    def calculate(cls, workers: List[Worker]) -> 'WageStatistics':
        """
        Calculate wage statistics from a list of workers.
        
        Args:
            workers: List of Worker objects
            
        Returns:
            WageStatistics object with calculated statistics
        """
        if not workers:
            return cls(0, 0, 0, 0.0, 0.0, 0.0, 0.0)
        
        permanent_workers = [w for w in workers if w.is_permanent]
        temporary_workers = [w for w in workers if w.is_temporary]
        
        total_wage_bill = sum(w.wage for w in workers)
        avg_wage_total = total_wage_bill / len(workers)
        
        avg_wage_permanent = (
            sum(w.wage for w in permanent_workers) / len(permanent_workers)
            if permanent_workers else 0.0
        )
        
        avg_wage_temporary = (
            sum(w.wage for w in temporary_workers) / len(temporary_workers)
            if temporary_workers else 0.0
        )
        
        return cls(
            total_workers=len(workers),
            permanent_workers=len(permanent_workers),
            temporary_workers=len(temporary_workers),
            avg_wage_total=avg_wage_total,
            avg_wage_permanent=avg_wage_permanent,
            avg_wage_temporary=avg_wage_temporary,
            total_wage_bill=total_wage_bill
        )

@dataclass
class NationalityStatistics:
    """
    Container for nationality distribution statistics (FROM SPEC-4).
    """
    total_workers: int
    permanent_nationalities: Dict[str, int]
    temporary_nationalities: Dict[str, int]
    
    @classmethod
    def calculate(cls, workers: List[Worker]) -> 'NationalityStatistics':
        """
        Calculate nationality statistics from a list of workers.
        
        Args:
            workers: List of Worker objects
            
        Returns:
            NationalityStatistics object with calculated statistics
        """
        if not workers:
            return cls(0, {}, {})
        
        permanent_nationalities = {}
        temporary_nationalities = {}
        
        for worker in workers:
            if worker.is_permanent:
                permanent_nationalities[worker.nationality] = permanent_nationalities.get(worker.nationality, 0) + 1
            else:
                temporary_nationalities[worker.nationality] = temporary_nationalities.get(worker.nationality, 0) + 1
        
        return cls(
            total_workers=len(workers),
            permanent_nationalities=permanent_nationalities,
            temporary_nationalities=temporary_nationalities
        )
    
    def get_temporary_distribution(self) -> Dict[str, float]:
        """
        Get temporary worker nationality distribution as proportions.
        
        Returns:
            Dictionary mapping nationality to proportion of temporary workers
        """
        total_temp = sum(self.temporary_nationalities.values())
        if total_temp == 0:
            return {}
        
        return {
            nationality: count / total_temp 
            for nationality, count in self.temporary_nationalities.items()
        }
    
    def get_top_temporary_nationalities(self, n: int = 3) -> Dict[str, float]:
        """
        Get top N nationalities for temporary workers.
        
        Args:
            n: Number of top nationalities to return
            
        Returns:
            Dictionary with top N nationalities and their proportions
        """
        distribution = self.get_temporary_distribution()
        sorted_nationalities = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_nationalities[:n])

@dataclass
class CountryCapStatistics:
    """
    Container for per-country cap statistics (FROM SPEC-5).
    """
    total_conversions: int
    conversions_by_country: Dict[str, int]
    queue_backlogs: Dict[str, int]
    per_country_limit: int
    cap_enabled: bool
    
    @classmethod
    def calculate(cls, country_queues: Dict[str, Deque[TemporaryWorker]], 
                 conversions_by_country: Dict[str, int], 
                 per_country_limit: int, cap_enabled: bool) -> 'CountryCapStatistics':
        """
        Calculate per-country cap statistics.
        
        Args:
            country_queues: Dictionary mapping nationality to conversion queue
            conversions_by_country: Conversions by nationality this year
            per_country_limit: Maximum conversions per country this year
            cap_enabled: Whether per-country cap is active
            
        Returns:
            CountryCapStatistics object
        """
        total_conversions = sum(conversions_by_country.values())
        queue_backlogs = {
            nationality: len(queue) 
            for nationality, queue in country_queues.items()
        }
        
        return cls(
            total_conversions=total_conversions,
            conversions_by_country=conversions_by_country.copy(),
            queue_backlogs=queue_backlogs,
            per_country_limit=per_country_limit,
            cap_enabled=cap_enabled
        )
    
    def get_utilization_by_country(self) -> Dict[str, float]:
        """
        Calculate cap utilization rate by country.
        
        Returns:
            Dictionary mapping nationality to utilization rate (0.0 to 1.0)
        """
        if not self.cap_enabled or self.per_country_limit == 0:
            return {}
        
        return {
            nationality: conversions / self.per_country_limit
            for nationality, conversions in self.conversions_by_country.items()
            if conversions > 0
        }
    
    def get_countries_at_cap(self) -> List[str]:
        """
        Get list of countries that hit their per-country cap.
        
        Returns:
            List of nationality strings that reached the cap
        """
        if not self.cap_enabled:
            return []
        
        return [
            nationality for nationality, conversions in self.conversions_by_country.items()
            if conversions >= self.per_country_limit
        ]

@dataclass
class BacklogAnalysis:
    """
    Container for backlog analysis statistics (NEW FOR SPEC-7).
    """
    scenario: str  # "capped" or "uncapped"
    backlog_by_nationality: Dict[str, int]
    total_backlog: int
    final_year: int
    
    @classmethod
    def from_simulation(cls, sim, scenario: str) -> 'BacklogAnalysis':
        """
        Create BacklogAnalysis from a completed simulation.
        
        Args:
            sim: Simulation object
            scenario: Scenario name ("capped" or "uncapped")
            
        Returns:
            BacklogAnalysis object
        """
        from .empirical_params import TEMP_NATIONALITY_DISTRIBUTION
        
        # Initialize all nationalities with zero backlog
        backlog_by_nationality = {nationality: 0 for nationality in TEMP_NATIONALITY_DISTRIBUTION.keys()}
        
        # Get actual backlogs from simulation queues
        if sim.country_cap_enabled and sim.country_queues:
            # Capped mode: get from country-specific queues
            for nationality, queue in sim.country_queues.items():
                backlog_by_nationality[nationality] = len(queue)
        elif sim.global_queue is not None:
            # Uncapped mode: count by nationality in global queue
            for temp_worker in sim.global_queue:
                if temp_worker.nationality in backlog_by_nationality:
                    backlog_by_nationality[temp_worker.nationality] += 1
        
        total_backlog = sum(backlog_by_nationality.values())
        final_year = sim.states[-1].year if sim.states else 2025
        
        return cls(
            scenario=scenario,
            backlog_by_nationality=backlog_by_nationality,
            total_backlog=total_backlog,
            final_year=final_year
        )
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert backlog analysis to pandas DataFrame for visualization.
        
        Returns:
            DataFrame with nationality, backlog_size, and scenario columns
        """
        import pandas as pd
        
        data = []
        for nationality, backlog_size in self.backlog_by_nationality.items():
            data.append({
                'nationality': nationality,
                'backlog_size': backlog_size,
                'scenario': self.scenario
            })
        
        return pd.DataFrame(data)
    
    def get_top_backlogs(self, n: int = 5) -> Dict[str, int]:
        """
        Get top N nationalities by backlog size.
        
        Args:
            n: Number of top nationalities to return
            
        Returns:
            Dictionary with top N nationalities and their backlog sizes
        """
        sorted_backlogs = sorted(self.backlog_by_nationality.items(), 
                               key=lambda x: x[1], reverse=True)
        return dict(sorted_backlogs[:n])
