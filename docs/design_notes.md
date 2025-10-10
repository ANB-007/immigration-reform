# Create design_notes.md
design_notes_content = '''# Design Notes - Workforce Growth Simulation

## Architecture Overview

The simulation follows a modular architecture with clear separation of concerns:

### Core Components

1. **Models (`models.py`)**: Data structures and domain objects
   - `Worker`: Individual agent representation
   - `SimulationState`: Aggregate state at each time step
   - `SimulationConfig`: Configuration parameters

2. **Simulation Engine (`sim.py`)**: Core business logic
   - `Simulation`: Main simulation class
   - Aggregate-based computation for performance
   - Hooks for future agent-based modeling

3. **CLI Interface (`cli.py`)**: User interaction
   - Argument parsing and validation
   - Progress reporting and results display
   - Integration with live data fetching

4. **Utilities (`utils.py`)**: Supporting functions
   - I/O operations (CSV save/load)
   - Data validation and formatting
   - Live data fetching capabilities

5. **Parameters (`empirical_params.py`)**: Configuration
   - Centralized empirical constants
   - Data source documentation
   - Validation ranges

## Design Decisions

### Aggregate vs. Agent-Based Modeling

**Current Implementation**: Aggregate-based for performance
- Tracks counts of worker types rather than individual agents
- Enables simulation of large populations (millions of workers)
- O(years) time complexity instead of O(workers × years)

**Future Extension**: Agent-based capabilities
- `Worker` dataclass ready for individual agent properties
- `to_agent_model()` method for conversion
- Serialization utilities for persistence

### Rounding Strategy

**Consistent Rounding**: Uses `round()` for all fractional calculations
- Ensures deterministic results
- Handles edge cases with small populations
- Documents strategy in specification compliance

**Alternative Considered**: `math.floor()` was considered but `round()` provides better accuracy for growth calculations.

### Error Handling

**Graceful Degradation**:
- Live data fetch failures fall back to defaults
- Configuration validation with helpful error messages
- Warnings for edge cases (small populations)

**Fail-Fast Validation**:
- Input validation at configuration time
- Data model validation in constructors
- Clear error messages for debugging

### Extensibility Hooks

**Future Features**:
- Skills and occupation modeling
- Employer-specific dynamics
- Geographic distribution
- Economic impact calculations

**Extension Points**:
- `Worker.attributes` dictionary for custom properties
- Pluggable growth rate functions
- Configurable rounding strategies
- Additional output formats

## Performance Considerations

### Memory Usage
- Aggregate modeling: O(years) memory usage
- Agent modeling: O(workers) memory usage
- Configurable simulation length limits

### Computational Complexity
- Current: O(years) time complexity
- Agent-based: O(workers × years) time complexity
- Scalability tested up to 1M initial workers

### I/O Optimization
- Streaming CSV output for large datasets
- Batched serialization for agent persistence
- Configurable output formatting

## Testing Strategy

### Unit Test Coverage
- Core simulation logic: `test_simulation.py`
- Utility functions: `test_utils.py`
- Data model validation: included in respective test files

### Test Categories
1. **Functionality Tests**: Core algorithm correctness
2. **Validation Tests**: Input validation and error handling
3. **Reproducibility Tests**: Seeded random number generation
4. **Edge Case Tests**: Small populations, extreme parameters

### Integration Testing
- End-to-end CLI execution
- File I/O operations
- Configuration validation

## Data Sources and Validation

### Primary Sources
1. **BLS Employment Data**: Labor force size and participation rates
2. **USCIS H-1B Data**: Visa approvals and holder estimates  
3. **Academic Research**: Immigration economic impact studies

### Data Quality
- Authoritative government sources preferred
- Timestamped data with source citations
- Fallback defaults for unavailable data
- Validation ranges for parameter sanity checking

### Update Frequency
- Annual parameter review recommended
- Live data fetching for current statistics
- Version control for parameter changes

## Security Considerations

### Data Privacy
- No personally identifiable information processed
- Aggregate statistics only
- Open data sources

### Network Security
- HTTPS-only for live data fetching
- Request timeout and retry logic
- Graceful handling of network failures

### Input Validation
- Parameter range checking
- File path validation
- Command injection prevention

## Future Enhancements

### Short Term
- Additional output formats (JSON, Excel)
- Graphical result visualization
- Parameter sensitivity analysis

### Medium Term
- Agent-based modeling capabilities
- Geographic distribution modeling
- Economic impact calculations

### Long Term
- Web interface for simulation
- Database integration for large-scale studies
- Machine learning parameter optimization

## Maintenance Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Comprehensive docstrings for all public functions
- Type hints for better IDE support

### Documentation
- Keep README.md updated with new features
- Document parameter changes with sources
- Maintain design decision rationale

### Version Control
- Tag releases with semantic versioning
- Document breaking changes
- Maintain backward compatibility where possible

## Known Limitations

### Current Constraints
1. **Static Growth Rates**: Parameters don't change over time
2. **No Economic Cycles**: No modeling of recessions/booms
3. **Homogeneous Workers**: All workers treated identically
4. **No Exits**: No modeling of retirement or emigration

### Mitigation Strategies
- Document assumptions clearly
- Provide parameter modification guidance
- Design extensibility hooks for future enhancements
- Validate results against historical data where possible

## References

### Technical
- NumPy documentation for random number generation
- Pandas documentation for data handling
- pytest documentation for testing patterns

### Domain Knowledge
- BLS methodology for labor force projections
- USCIS H-1B program documentation
- Immigration economics research literature
