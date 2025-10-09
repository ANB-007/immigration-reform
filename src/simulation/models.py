# src/simulation/models.py
"""
Data models for the workforce simulation.
Defines Worker agents and related data structures.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

class WorkerStatus(Enum):
    """Enumeration of worker status types."""
    PERMANENT = "permanent"
    TEMPORARY = "temporary"  # H-1B holders

@dataclass
class Worker:
    """
    Represents a single worker agent in the simulation.
    
    Attributes:
        id: Unique identifier for the worker
        status: Worker status (permanent or temporary/H-1B)
        age: Age of the worker in years
        skills: Skill level or category (for future extensions)
        occupation: Job category or title (for future extensions)
        employer_id: ID of employing organization (for future extensions)
        entry_year: Year the worker entered the workforce
        year_joined: Year the worker joined as temporary (for FIFO conversion)
        attributes: Additional extensible attributes
    """
    id: int
    status: WorkerStatus
    age: int = 35  # Default age
    entry_year: int = 2025  # Default entry year
    year_joined: int = 2025  # Year joined as temporary worker (NEW FOR SPEC-2)
    skills: Optional[str] = None
    occupation: Optional[str] = None
    employer_id: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.age < 16 or self.age > 100:
            raise ValueError(f"Invalid age: {self.age}. Must be between 16 and 100.")
        
        # Set year_joined to entry_year if not specified and worker is temporary
        if self.status == WorkerStatus.TEMPORARY and self.year_joined == 2025 and self.entry_year != 2025:
            self.year_joined = self.entry_year
    
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
        """Calculate years the worker has been in temporary status (NEW FOR SPEC-2)."""
        if self.status != WorkerStatus.TEMPORARY:
            return 0
        return max(0, current_year - self.year_joined)

@dataclass
class SimulationState:
    """
    Represents the state of the simulation at a given time step.
    
    Attributes:
        year: Current simulation year
        total_workers: Total number of workers
        permanent_workers: Number of permanent workers
        temporary_workers: Number of temporary/H-1B workers
        new_permanent: New permanent workers added this year
        new_temporary: New temporary workers added this year
        converted_temps: Temporary workers converted to permanent (NEW FOR SPEC-2)
    """
    year: int
    total_workers: int
    permanent_workers: int
    temporary_workers: int
    new_permanent: int = 0
    new_temporary: int = 0
    converted_temps: int = 0  # NEW FOR SPEC-2
    
    def __post_init__(self):
        """Validation of state consistency."""
        if self.permanent_workers + self.temporary_workers != self.total_workers:
            raise ValueError("Sum of permanent and temporary workers must equal total workers")
        
        if any(count < 0 for count in [self.total_workers, self.permanent_workers, 
                                     self.temporary_workers, self.new_permanent, 
                                     self.new_temporary, self.converted_temps]):
            raise ValueError("All worker counts must be non-negative")

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
    
    def __post_init__(self):
        """Validation of configuration parameters."""
        if self.initial_workers <= 0:
            raise ValueError("Initial workers must be positive")
        if self.years <= 0:
            raise ValueError("Simulation years must be positive")

@dataclass
class TemporaryWorker:
    """
    Represents a temporary worker in the conversion queue (NEW FOR SPEC-2).
    Used for FIFO ordering of temporary-to-permanent conversions.
    """
    worker_id: int
    year_joined: int
    
    def __lt__(self, other):
        """Less than comparison for sorting (earlier year_joined comes first)."""
        return self.year_joined < other.year_joined
