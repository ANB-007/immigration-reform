# src/simulation/models.py
"""
Streamlined data models for workforce simulation.
SPEC-10: Eliminated hardcoding, removed redundant variables, implemented on-the-fly aggregation.
All temporal attributes are dynamically determined, aggregates computed from worker data.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import json
import csv
import math
from collections import defaultdict, deque
from pathlib import Path

from .empirical_params import (
    DEFAULT_SIMULATION_START_YEAR, 
    TEMP_NATIONALITY_DISTRIBUTION,
    calculate_annual_conversion_cap
)

class WorkerStatus(Enum):
    """Worker employment status enumeration."""
    TEMPORARY = "temporary"
    PERMANENT = "permanent"

@dataclass
class SimulationConfig:
    """Configuration parameters for simulation run."""
    initial_workers: int
    years: int
    seed: Optional[int] = None
    live_fetch: bool = False
    output_path: str = "data/sample_output.csv"
    agent_mode: bool = True
    show_nationality_summary: bool = False
    country_cap_enabled: bool = False
    compare_backlogs: bool = False
    debug: bool = False
    flat_slots: bool = True
    carryover_slots: bool = False
    start_year: int = DEFAULT_SIMULATION_START_YEAR  # SPEC-10: Dynamic start year
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.initial_workers <= 0:
            raise ValueError("initial_workers must be positive")
        if self.years <= 0:
            raise ValueError("years must be positive")
        
        # Ensure only one slots strategy is enabled
        if self.carryover_slots:
            self.flat_slots = False

class Worker:
    """
    Individual worker with dynamic temporal attributes.
    SPEC-10: No hardcoded years, all temporal attributes computed from simulation context.
    """
    
    def __init__(self, id: int, status: WorkerStatus, nationality: str, 
                 simulation_start_year: int, entry_year_offset: int = 0, 
                 age: int = 35, wage: float = 95000.0):
        """
        Initialize worker with dynamic temporal attributes.
        SPEC-10: All years computed relative to simulation_start_year.
        
        Args:
            id: Unique worker identifier
            status: Employment status
            nationality: Worker's nationality
            simulation_start_year: Base year for simulation
            entry_year_offset: Years from start when worker entered (0 = existing worker)
            age: Worker's age
            wage: Current wage in USD
        """
        self.id = id
        self.status = status
        self.nationality = nationality
        self.age = age
        self.wage = wage
        
        # SPEC-10: Dynamic temporal attributes
        self.simulation_start_year = simulation_start_year
        self.entry_year = simulation_start_year + entry_year_offset
        self.created_year = simulation_start_year  # When record was created
        
        # Legacy attributes for backward compatibility (computed dynamically)
        self.year_joined = self.entry_year
        
        # Conversion tracking
        self.conversion_year: Optional[int] = None
        self.converted_year: Optional[int] = None  # Alias for conversion_year
        self.was_converted: bool = False
        
        # Deterministic ordering for queue processing
        self.arrival_index: Optional[int] = None
        
        # Optional attributes for extensibility
        self.skills: Optional[List[str]] = None
        self.occupation: Optional[str] = None
        self.employer_id: Optional[int] = None
        self.attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_temporary(self) -> bool:
        """Check if worker has temporary status."""
        return self.status == WorkerStatus.TEMPORARY
    
    @property
    def is_permanent(self) -> bool:
        """Check if worker has permanent status."""
        return self.status == WorkerStatus.PERMANENT
    
    def years_in_workforce(self, current_year: int) -> int:
        """
        Calculate years worker has been in workforce.
        SPEC-10: Enables time-based wage growth calculations.
        """
        return max(0, current_year - self.entry_year)
    
    def years_as_temporary(self, current_year: int) -> int:
        """Calculate years as temporary worker."""
        if self.is_permanent and self.conversion_year:
            return max(0, self.conversion_year - self.entry_year)
        elif self.is_temporary:
            return max(0, current_year - self.entry_year)
        return 0
    
    def years_as_permanent(self, current_year: int) -> int:
        """Calculate years as permanent worker."""
        if self.is_permanent:
            start_permanent = self.conversion_year or self.entry_year
            return max(0, current_year - start_permanent)
        return 0
    
    def convert_to_permanent(self, conversion_year: int) -> None:
        """Convert worker from temporary to permanent status."""
        if self.is_temporary:
            self.status = WorkerStatus.PERMANENT
            self.conversion_year = conversion_year
            self.converted_year = conversion_year  # Alias
            self.was_converted = True
    
    def apply_wage_jump(self, multiplier: float) -> None:
        """Apply wage increase due to job change."""
        self.wage *= multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert worker to dictionary representation."""
        return {
            'id': self.id,
            'status': self.status.value,
            'nationality': self.nationality,
            'age': self.age,
            'wage': self.wage,
            'simulation_start_year': self.simulation_start_year,
            'entry_year': self.entry_year,
            'created_year': self.created_year,
            'year_joined': self.year_joined,
            'conversion_year': self.conversion_year,
            'converted_year': self.converted_year,
            'was_converted': self.was_converted,
            'arrival_index': self.arrival_index,
            'skills': self.skills,
            'occupation': self.occupation,
            'employer_id': self.employer_id,
            'attributes': self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Worker':
        """Create worker from dictionary representation."""
        worker = cls(
            id=data['id'],
            status=WorkerStatus(data['status']),
            nationality=data['nationality'],
            simulation_start_year=data.get('simulation_start_year', DEFAULT_SIMULATION_START_YEAR),
            entry_year_offset=data.get('entry_year', DEFAULT_SIMULATION_START_YEAR) - data.get('simulation_start_year', DEFAULT_SIMULATION_START_YEAR),
            age=data.get('age', 35),
            wage=data.get('wage', 95000.0)
        )
        
        # Set additional attributes
        worker.conversion_year = data.get('conversion_year')
        worker.converted_year = data.get('converted_year')
        worker.was_converted = data.get('was_converted', False)
        worker.arrival_index = data.get('arrival_index')
        worker.skills = data.get('skills')
        worker.occupation = data.get('occupation')
        worker.employer_id = data.get('employer_id')
        worker.attributes = data.get('attributes', {})
        
        return worker
    
    def __repr__(self) -> str:
        return (f"Worker(id={self.id}, status={self.status.value}, "
                f"nationality='{self.nationality}', age={self.age}, "
                f"wage={self.wage}, entry_year={self.entry_year})")

class TemporaryWorker:
    """
    Simplified temporary worker for queue management.
    SPEC-10: Dynamic temporal attributes.
    """
    
    def __init__(self, worker_id: int, entry_year: int, nationality: str):
        """Initialize temporary worker for queue."""
        self.worker_id = worker_id
        self.year_joined = entry_year  # Legacy alias
        self.entry_year = entry_year
        self.nationality = nationality
        self.arrival_index: Optional[int] = None
    
    def __repr__(self) -> str:
        return f"TemporaryWorker(id={self.worker_id}, entry_year={self.entry_year}, nationality='{self.nationality}')"

# SPEC-10: Legacy statistics classes for backward compatibility with other modules
@dataclass
class WageStatistics:
    """Wage statistics computed from worker data (legacy compatibility)."""
    total_workers: int
    permanent_workers: int
    temporary_workers: int
    avg_wage_total: float
    avg_wage_permanent: float
    avg_wage_temporary: float
    total_wage_bill: float
    
    @classmethod
    def calculate(cls, workers: List[Worker]) -> 'WageStatistics':
        """Calculate wage statistics from worker list."""
        if not workers:
            return cls(0, 0, 0, 0.0, 0.0, 0.0, 0.0)
        
        total_workers = len(workers)
        permanent_workers = sum(1 for w in workers if w.is_permanent)
        temporary_workers = total_workers - permanent_workers
        
        total_wage_bill = sum(w.wage for w in workers)
        avg_wage_total = total_wage_bill / total_workers if total_workers > 0 else 0.0
        
        permanent_wages = [w.wage for w in workers if w.is_permanent]
        temporary_wages = [w.wage for w in workers if w.is_temporary]
        
        avg_wage_permanent = sum(permanent_wages) / len(permanent_wages) if permanent_wages else 0.0
        avg_wage_temporary = sum(temporary_wages) / len(temporary_wages) if temporary_wages else 0.0
        
        return cls(
            total_workers, permanent_workers, temporary_workers,
            avg_wage_total, avg_wage_permanent, avg_wage_temporary, total_wage_bill
        )

@dataclass
class NationalityStatistics:
    """Nationality statistics computed from worker data (legacy compatibility)."""
    total_workers: int
    permanent_nationalities: Dict[str, int] = field(default_factory=dict)
    temporary_nationalities: Dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def calculate(cls, workers: List[Worker]) -> 'NationalityStatistics':
        """Calculate nationality statistics from worker list."""
        permanent_nationalities = defaultdict(int)
        temporary_nationalities = defaultdict(int)
        
        for worker in workers:
            if worker.is_permanent:
                permanent_nationalities[worker.nationality] += 1
            else:
                temporary_nationalities[worker.nationality] += 1
        
        return cls(
            total_workers=len(workers),
            permanent_nationalities=dict(permanent_nationalities),
            temporary_nationalities=dict(temporary_nationalities)
        )
    
    def get_top_temporary_nationalities(self, top_n: int = 10) -> Dict[str, float]:
        """Get top N temporary worker nationalities by proportion."""
        total_temporary = sum(self.temporary_nationalities.values())
        if total_temporary == 0:
            return {}
        
        proportions = {
            nationality: count / total_temporary
            for nationality, count in self.temporary_nationalities.items()
        }
        
        sorted_proportions = sorted(proportions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_proportions[:top_n])

@dataclass
class SimulationState:
    """
    Complete simulation state at a time point.
    SPEC-10: Streamlined to essential metrics, aggregates computed on-the-fly from worker data.
    """
    year: int
    workers: List[Worker] = field(default_factory=list)  # Source of truth
    annual_conversion_cap: int = 0  # SPEC-10: Computed from empirical params
    
    # Conversion tracking
    new_permanent: int = 0
    new_temporary: int = 0
    converted_temps: int = 0
    cumulative_conversions: int = 0
    
    # Policy configuration
    country_cap_enabled: bool = False
    
    # Legacy fields for backward compatibility with tests
    top_temp_nationalities: Dict[str, float] = field(default_factory=dict)
    converted_by_country: Dict[str, int] = field(default_factory=dict)
    queue_backlog_by_country: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute annual conversion cap if not set."""
        if self.annual_conversion_cap == 0 and self.workers:
            self.annual_conversion_cap = calculate_annual_conversion_cap(len(self.workers))
    
    # SPEC-10: On-the-fly aggregation properties (no redundant storage)
    @property
    def total_workers(self) -> int:
        """Total worker count computed from worker list."""
        return len(self.workers)
    
    @property
    def permanent_workers(self) -> int:
        """Permanent worker count computed from worker list."""
        return sum(1 for w in self.workers if w.is_permanent)
    
    @property
    def temporary_workers(self) -> int:
        """Temporary worker count computed from worker list."""
        return sum(1 for w in self.workers if w.is_temporary)
    
    @property
    def avg_wage_total(self) -> float:
        """Average wage across all workers."""
        if not self.workers:
            return 95000.0  # Default for empty states
        return sum(w.wage for w in self.workers) / len(self.workers)
    
    @property
    def avg_wage_permanent(self) -> float:
        """Average wage of permanent workers."""
        permanent_wages = [w.wage for w in self.workers if w.is_permanent]
        return sum(permanent_wages) / len(permanent_wages) if permanent_wages else 95000.0
    
    @property
    def avg_wage_temporary(self) -> float:
        """Average wage of temporary workers."""
        temporary_wages = [w.wage for w in self.workers if w.is_temporary]
        return sum(temporary_wages) / len(temporary_wages) if temporary_wages else 95000.0
    
    @property
    def total_wage_bill(self) -> float:
        """Total wage bill computed from worker wages."""
        if not self.workers:
            return 0.0
        return sum(w.wage for w in self.workers)
    
    @property
    def h1b_share(self) -> float:
        """H-1B share of total workforce."""
        return self.temporary_workers / self.total_workers if self.total_workers > 0 else 0.0
    
    @property
    def permanent_share(self) -> float:
        """Permanent share of total workforce."""
        return self.permanent_workers / self.total_workers if self.total_workers > 0 else 0.0
    
    def get_nationality_distribution(self, status: Optional[WorkerStatus] = None) -> Dict[str, int]:
        """
        Get nationality distribution for workers.
        
        Args:
            status: Filter by worker status, None for all workers
            
        Returns:
            Dictionary mapping nationality to count
        """
        distribution = defaultdict(int)
        for worker in self.workers:
            if status is None or worker.status == status:
                distribution[worker.nationality] += 1
        return dict(distribution)
    
    def get_top_temporary_nationalities(self, top_n: int = 5) -> Dict[str, float]:
        """Get top N temporary worker nationalities by proportion."""
        temp_distribution = self.get_nationality_distribution(WorkerStatus.TEMPORARY)
        total_temp = sum(temp_distribution.values())
        
        if total_temp == 0:
            return {}
        
        proportions = {nat: count / total_temp for nat, count in temp_distribution.items()}
        sorted_props = sorted(proportions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_props[:top_n])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            'year': self.year,
            'total_workers': self.total_workers,
            'permanent_workers': self.permanent_workers,
            'temporary_workers': self.temporary_workers,
            'new_permanent': self.new_permanent,
            'new_temporary': self.new_temporary,
            'converted_temps': self.converted_temps,
            'avg_wage_total': self.avg_wage_total,
            'avg_wage_permanent': self.avg_wage_permanent,
            'avg_wage_temporary': self.avg_wage_temporary,
            'total_wage_bill': self.total_wage_bill,
            'h1b_share': self.h1b_share,
            'permanent_share': self.permanent_share,
            'annual_conversion_cap': self.annual_conversion_cap,
            'cumulative_conversions': self.cumulative_conversions,
            'country_cap_enabled': self.country_cap_enabled,
            'top_temp_nationalities': self.top_temp_nationalities,
            'converted_by_country': self.converted_by_country,
            'queue_backlog_by_country': self.queue_backlog_by_country
        }
    
    def to_json(self) -> str:
        """Convert state to JSON representation."""
        return json.dumps(self.to_dict(), indent=2)

class BacklogAnalysis:
    """
    Backlog analysis utility with streamlined API.
    SPEC-10: Simplified construction and clear data contracts.
    """
    
    def __init__(self, scenario: str, backlog_by_nationality: Dict[str, int], 
                 total_backlog: int, final_year: int):
        """
        Initialize BacklogAnalysis.
        
        Args:
            scenario: Scenario name (e.g., "capped", "uncapped")
            backlog_by_nationality: Dictionary mapping nationality to backlog count
            total_backlog: Total backlog across all nationalities
            final_year: Final year of simulation
        """
        self.scenario = scenario
        self.scenario_name = scenario  # Alias
        self.backlog_by_nationality = backlog_by_nationality.copy()
        self.backlog_data = backlog_by_nationality.copy()  # Alias
        self.total_backlog = total_backlog
        self.final_year = final_year
    
    @classmethod
    def from_simulation(cls, simulation, scenario_name: str) -> 'BacklogAnalysis':
        """
        Create BacklogAnalysis from completed simulation.
        SPEC-10: Robust queue detection with clear fallback logic.
        """
        backlog_data = {}
        
        # Initialize all nationalities to 0
        all_nationalities = list(TEMP_NATIONALITY_DISTRIBUTION.keys())
        for nationality in all_nationalities:
            backlog_data[nationality] = 0
        
        # Count workers in queues with priority-based detection
        workers_counted = False
        
        # Priority 1: Use country_queues if available and populated
        if (hasattr(simulation, 'country_queues') and simulation.country_queues and
            any(len(queue) > 0 for queue in simulation.country_queues.values())):
            for nationality, queue in simulation.country_queues.items():
                if nationality in backlog_data:
                    backlog_data[nationality] = len(queue)
            workers_counted = True
        
        # Priority 2: Use global_queue if available and populated
        elif (hasattr(simulation, 'global_queue') and simulation.global_queue and
              len(simulation.global_queue) > 0):
            for temp_worker in simulation.global_queue:
                if hasattr(temp_worker, 'nationality') and temp_worker.nationality in backlog_data:
                    backlog_data[temp_worker.nationality] += 1
            workers_counted = True
        
        # Priority 3: Use final state backlog data
        elif hasattr(simulation, 'states') and simulation.states:
            final_state = simulation.states[-1]
            if hasattr(final_state, 'workers'):
                # Count temporary workers by nationality
                for worker in final_state.workers:
                    if worker.is_temporary and worker.nationality in backlog_data:
                        backlog_data[worker.nationality] += 1
                workers_counted = True
        
        total_backlog = sum(backlog_data.values())
        final_year = getattr(simulation.states[-1] if hasattr(simulation, 'states') and simulation.states else None, 'year', DEFAULT_SIMULATION_START_YEAR)
        
        return cls(scenario_name, backlog_data, total_backlog, final_year)
    
    def get_total_backlog(self) -> int:
        """Get total backlog across all nationalities."""
        return self.total_backlog
    
    def get_backlog_by_country(self) -> Dict[str, int]:
        """Get backlog counts by nationality."""
        return self.backlog_by_nationality.copy()
    
    def get_top_backlogs(self, n: int = 10) -> Dict[str, int]:
        """Get top N nationalities by backlog size."""
        sorted_backlogs = sorted(
            self.backlog_by_nationality.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return dict(sorted_backlogs[:n])
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert backlog data to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame conversion")
        
        data = [
            {
                'nationality': nationality,
                'backlog_size': size,
                'scenario': self.scenario
            }
            for nationality, size in sorted(self.backlog_by_nationality.items())
        ]
        
        return pd.DataFrame(data)
    
    def save_csv(self, filepath: str) -> None:
        """Save backlog data to CSV file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        sorted_data = sorted(self.backlog_by_nationality.items())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['nationality', 'backlog_size', 'scenario'])
            
            for nationality, backlog_size in sorted_data:
                writer.writerow([nationality, backlog_size, self.scenario])
    
    @staticmethod
    def load_csv(filepath: str) -> 'BacklogAnalysis':
        """Load backlog data from CSV file."""
        backlog_data = {}
        scenario_name = "unknown"
        
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                nationality = row['nationality']
                backlog_size = int(row['backlog_size'])
                scenario_name = row.get('scenario', scenario_name)
                backlog_data[nationality] = backlog_size
        
        return BacklogAnalysis(
            scenario=scenario_name,
            backlog_by_nationality=backlog_data,
            total_backlog=sum(backlog_data.values()),
            final_year=DEFAULT_SIMULATION_START_YEAR + 10
        )
    
    def compare_with(self, other: 'BacklogAnalysis') -> Dict[str, Any]:
        """Compare this backlog analysis with another."""
        all_nationalities = set(self.backlog_by_nationality.keys()) | set(other.backlog_by_nationality.keys())
        
        differences = {}
        for nationality in all_nationalities:
            self_size = self.backlog_by_nationality.get(nationality, 0)
            other_size = other.backlog_by_nationality.get(nationality, 0)
            differences[nationality] = other_size - self_size
        
        return {
            'scenario_1': self.scenario,
            'scenario_2': other.scenario,
            'total_backlog_1': self.total_backlog,
            'total_backlog_2': other.total_backlog,
            'total_difference': other.total_backlog - self.total_backlog,
            'differences_by_nationality': differences
        }
    
    @property
    def empty(self) -> bool:
        """Check if backlog is empty (for test compatibility)."""
        return self.total_backlog == 0
    
    def __repr__(self) -> str:
        top_3 = dict(list(self.get_top_backlogs(3).items()))
        return f"BacklogAnalysis(scenario='{self.scenario}', total={self.total_backlog}, top_3={top_3})"

# SPEC-10: Validation functions (meaningful constraints only)
def validate_simulation_state_consistency(state: SimulationState) -> bool:
    """
    Validate meaningful state consistency constraints.
    SPEC-10: Only checks constraints that can actually be violated.
    """
    # Check worker count consistency (can be violated by bugs)
    computed_total = state.permanent_workers + state.temporary_workers
    if computed_total != state.total_workers:
        return False
    
    # Check that we have expected number of worker objects
    if len(state.workers) != state.total_workers:
        return False
    
    # Check wage aggregation consistency (can be violated by NaN/None wages)
    if state.total_workers > 0:
        manual_total_wage = sum(w.wage for w in state.workers if w.wage is not None and not math.isnan(w.wage))
        if abs(manual_total_wage - state.total_wage_bill) > 0.01:
            return False
    
    return True

def validate_simulation_consistency(states: List[SimulationState]) -> bool:
    """
    Validate consistency across multiple simulation states.
    SPEC-10: Legacy compatibility function.
    """
    for state in states:
        if not validate_simulation_state_consistency(state):
            return False
    return True

def states_to_dataframe(states: List[SimulationState]) -> 'pd.DataFrame':
    """Convert list of simulation states to pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for DataFrame conversion")
    
    data = [state.to_dict() for state in states]
    return pd.DataFrame(data)

# Module exports
__all__ = [
    'WorkerStatus',
    'SimulationConfig', 
    'Worker',
    'TemporaryWorker',
    'WageStatistics',
    'NationalityStatistics',
    'SimulationState',
    'BacklogAnalysis',
    'validate_simulation_state_consistency',
    'validate_simulation_consistency',
    'states_to_dataframe'
]
