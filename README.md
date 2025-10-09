# Workforce Growth Simulation

A Python simulation modeling the growth dynamics of permanent versus temporary (H-1B) workers in the U.S. workforce over time, with temporary-to-permanent conversion capabilities.

## Overview

This simulation models how the U.S. workforce grows over time, specifically tracking the dynamics between permanent workers and temporary workers on H-1B visas. The model uses empirically-derived parameters from authoritative sources to ensure realistic projections. **NEW IN VERSION 2**: The simulation now includes a temporary-to-permanent conversion system that models the green card process with realistic annual caps.

## Features

- **Configurable Parameters**: All empirical rates and proportions are centralized in `empirical_params.py`
- **Proportional Growth**: Maintains realistic workforce composition ratios
- **Green Card Conversions**: Models temporary-to-permanent status transitions with annual caps (NEW)
- **FIFO Conversion Logic**: First-come-first-served conversion order for temporary workers (NEW)
- **Live Data Fetching**: Can fetch current statistics from authoritative sources (with `--live-fetch`)
- **Reproducible**: Uses seeded random number generation for consistent results
- **Extensible**: Designed for future expansion to agent-based modeling
- **Well-tested**: Comprehensive unit test coverage

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup
