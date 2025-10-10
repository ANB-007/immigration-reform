# src/simulation/models.py
"""
Data models for the workforce simulation.
Defines Worker agents and related data structures.
Updated for SPEC-8 to remove ConversionCapTracker (moved logic to Simulation class).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

from .empirical_params import STARTING_WAGE, PERMANENT_NATIONALITY

class WorkerStatus(Enum):
    """Enumeration of worker status types."""
    PERMANENT = "permanent"
    TEMPORARY = "temporary"  # H-1B holders

@dataclass
class Worker:
    """
    Represents a single worker agent in the simulation.
    Updated for SPEC-8 with enhanced wage tracking and status-dependent behavior.
    """
    id: int
    status: WorkerStatus
    nationality: str  # FROM SPEC-4: Immutable nationality
    age: int = 35  # Default age
    year_joined: int = 2025  # Year joined as temporary worker
    wage: float = STARTING_WAGE  # FROM SPEC-3: Current wage in USD per year
    created_year: int = 2025  # FROM SPEC-3: Year worker was created/hired
    entry_year: int = 2025  # Year entered workforce
    conversion_year: Optional[int] = None  # FROM SPEC-8: Year converted to permanent
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
    
    @property
    def was_converted(self) -> bool:
        """Returns True if worker was converted from temporary to permanent (FROM SPEC-8)."""
        return self.conversion_year is not None
    
    def years_in_workforce(self, current_year: int) -> int:
        """Calculate years the worker has been in the workforce."""
        return max(0, current_year - self.entry_year)
    
    def years_as_temporary(self, current_year: int) -> int:
        """Calculate years the worker has been in temporary status."""
        if self.status != WorkerStatus.TEMPORARY:
            return 0
        return max(0, current_year - self.year_joined)
    
    def years_as_permanent(self, current_year: int) -> int:
        """Calculate years the worker has been in permanent status (FROM SPEC-8)."""
        if self.status != WorkerStatus.PERMANENT:
            return 0
        if self.conversion_year is not None:
            # Converted worker: count from conversion year
            return max(0, current_year - self.conversion_year)
        else:
            # Initially permanent worker: count from entry year
            return max(0, current_year - self.entry_year)
    
    def apply_wage_jump(self, jump_factor: float) -> None:
        """Apply wage jump due to job change (FROM SPEC-3)."""
        if jump_factor < 0:
            raise ValueError(f"Jump factor must be non-negative, got {jump_factor}")
        
        # Ensure wage doesn't decrease (minimum factor of 1.0)
        effective_factor = max(jump_factor, 1.0)
        self.wage *= effective_factor
    
    def convert_to_permanent(self, conversion_year: int) -> None:
        """
        Convert worker from temporary to permanent status (FROM SPEC-4, UPDATED FOR SPEC-8).
        Nationality remains unchanged as per SPEC-4 requirements.
        """
        if self.status == WorkerStatus.TEMPORARY:
            self.status = WorkerStatus.PERMANENT
            self.conversion_year = conversion_year  # FROM SPEC-8
        else:
            raise ValueError(f"Cannot convert worker {self.id}: already permanent")

@dataclass
class SimulationState:
    """
    Represents the state of the simulation at a given time step.
    Updated for SPEC-8 to include fixed annual conversion tracking.
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
    annual_conversion_cap: int = 0  # FROM SPEC-8: Fixed annual cap
    cumulative_conversions: int = 0  # FROM SPEC-8: Running total of conversions
    
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
        
        # FROM SPEC-8: Validate conversion tracking
        if self.annual_conversion_cap < 0:
            raise ValueError("Annual conversion cap must be non-negative")
        
        if self.cumulative_conversions < 0:
            raise ValueError("Cumulative conversions must be non-negative")

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
    compare_backlogs: bool = False  # FROM SPEC-7: Enable backlog comparison analysis
    debug: bool = False  # FROM SPEC-8: Enable debug output
    
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
    """Container for wage statistics calculation (FROM SPEC-3)."""
    total_workers: int
    permanent_workers: int
    temporary_workers: int
    avg_wage_total: float
    avg_wage_permanent: float
    avg_wage_temporary: float
    total_wage_bill: float
    
    @classmethod
    def calculate(cls, workers: List[Worker]) -> 'WageStatistics':
        """Calculate wage statistics from a list of workers."""
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
    """Container for nationality distribution statistics (FROM SPEC-4)."""
    total_workers: int
    permanent_nationalities: Dict[str, int]
    temporary_nationalities: Dict[str, int]
    
    @classmethod
    def calculate(cls, workers: List[Worker]) -> 'NationalityStatistics':
        """Calculate nationality statistics from a list of workers."""
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
        """Get temporary worker nationality distribution as proportions."""
        total_temp = sum(self.temporary_nationalities.values())
        if total_temp == 0:
            return {}
        
        return {
            nationality: count / total_temp 
            for nationality, count in self.temporary_nationalities.items()
        }
    
    def get_top_temporary_nationalities(self, n: int = 3) -> Dict[str, float]:
        """Get top N nationalities for temporary workers."""
        distribution = self.get_temporary_distribution()
        sorted_nationalities = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_nationalities[:n])

@dataclass
class BacklogAnalysis:
    """Container for backlog analysis statistics (FROM SPEC-7)."""
    scenario: str  # "capped" or "uncapped"
    backlog_by_nationality: Dict[str, int]
    total_backlog: int
    final_year: int
    
    @classmethod
    def from_simulation(cls, sim, scenario: str) -> 'BacklogAnalysis':
        """Create BacklogAnalysis from a completed simulation."""
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
        """Convert backlog analysis to pandas DataFrame for visualization."""
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
        """Get top N nationalities by backlog size."""
        sorted_backlogs = sorted(self.backlog_by_nationality.items(), 
                               key=lambda x: x[1], reverse=True)
        return dict(sorted_backlogs[:n])
