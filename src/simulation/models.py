# src/simulation/models.py
"""
Data models for the workforce simulation.
Defines Worker agents and related data structures.
Updated for SPEC-3 to include wage tracking and job-to-job transitions.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

from .empirical_params import STARTING_WAGE

class WorkerStatus(Enum):
    """Enumeration of worker status types."""
    PERMANENT = "permanent"
    TEMPORARY = "temporary"  # H-1B holders

@dataclass
class Worker:
    """
    Represents a single worker agent in the simulation.
    Updated for SPEC-3 to include wage and creation year tracking.
    
    Attributes:
        id: Unique identifier for the worker
        status: Worker status (permanent or temporary/H-1B)
        age: Age of the worker in years
        year_joined: Year the worker joined as temporary (for FIFO conversion)
        wage: Current wage in USD per year (NEW FOR SPEC-3)
        created_year: Year the worker was created/hired (NEW FOR SPEC-3)
        entry_year: Year the worker entered the workforce
        skills: Skill level or category (for future extensions)
        occupation: Job category or title (for future extensions)
        employer_id: ID of employing organization (for future extensions)
        attributes: Additional extensible attributes
    """
    id: int
    status: WorkerStatus
    age: int = 35  # Default age
    year_joined: int = 2025  # Year joined as temporary worker
    wage: float = STARTING_WAGE  # NEW FOR SPEC-3: Current wage in USD per year
    created_year: int = 2025  # NEW FOR SPEC-3: Year worker was created/hired
    entry_year: int = 2025  # Year entered workforce
    skills: Optional[List[str]] = None  # Updated to List for SPEC-3
    occupation: Optional[str] = None
    employer_id: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.age < 16 or self.age > 100:
            raise ValueError(f"Invalid age: {self.age}. Must be between 16 and 100.")
        
        if self.wage < 0:
            raise ValueError(f"Invalid wage: {self.wage}. Must be non-negative.")
        
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
        Apply wage jump due to job change (NEW FOR SPEC-3).
        
        Args:
            jump_factor: Multiplicative factor for wage increase (e.g., 1.08 for 8% increase)
        """
        if jump_factor < 0:
            raise ValueError(f"Jump factor must be non-negative, got {jump_factor}")
        
        # Ensure wage doesn't decrease (minimum factor of 1.0)
        effective_factor = max(jump_factor, 1.0)
        self.wage *= effective_factor

@dataclass
class SimulationState:
    """
    Represents the state of the simulation at a given time step.
    Updated for SPEC-3 to include wage statistics.
    
    Attributes:
        year: Current simulation year
        total_workers: Total number of workers
        permanent_workers: Number of permanent workers
        temporary_workers: Number of temporary/H-1B workers
        new_permanent: New permanent workers added this year
        new_temporary: New temporary workers added this year
        converted_temps: Temporary workers converted to permanent
        avg_wage_total: Average wage across all workers (NEW FOR SPEC-3)
        avg_wage_permanent: Average wage for permanent workers (NEW FOR SPEC-3)
        avg_wage_temporary: Average wage for temporary workers (NEW FOR SPEC-3)
        total_wage_bill: Total wages paid to all workers (NEW FOR SPEC-3)
    """
    year: int
    total_workers: int
    permanent_workers: int
    temporary_workers: int
    new_permanent: int = 0
    new_temporary: int = 0
    converted_temps: int = 0
    avg_wage_total: float = 0.0  # NEW FOR SPEC-3
    avg_wage_permanent: float = 0.0  # NEW FOR SPEC-3
    avg_wage_temporary: float = 0.0  # NEW FOR SPEC-3
    total_wage_bill: float = 0.0  # NEW FOR SPEC-3
    
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
    agent_mode: bool = True  # NEW FOR SPEC-3: Default to agent-mode for wage tracking
    
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
    """
    worker_id: int
    year_joined: int
    
    def __lt__(self, other):
        """Less than comparison for sorting (earlier year_joined comes first)."""
        if self.year_joined != other.year_joined:
            return self.year_joined < other.year_joined
        # Use worker_id as tiebreaker for same year
        return self.worker_id < other.worker_id

@dataclass
class WageStatistics:
    """
    Container for wage statistics calculation (NEW FOR SPEC-3).
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
