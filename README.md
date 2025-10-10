# Workforce Growth Simulation

A Python simulation modeling the growth dynamics of permanent versus temporary (H-1B) workers in the U.S. workforce over time, with comprehensive wage tracking, nationality segmentation, per-country conversion caps, enhanced visualization capabilities, and comparative backlog analysis.

## Overview

This simulation models how the U.S. workforce grows over time, specifically tracking the dynamics between permanent workers and temporary workers on H-1B visas. The model uses empirically-derived parameters from authoritative sources to ensure realistic projections.

## Features

### Core Simulation (SPEC-1 & SPEC-2)
- **Configurable Parameters**: All empirical rates and proportions centralized in `empirical_params.py`
- **Proportional Growth**: Maintains realistic workforce composition ratios
- **Green Card Conversions**: Models temporary-to-permanent status transitions with annual caps
- **FIFO Conversion Logic**: First-come-first-served conversion order for temporary workers

### Advanced Analytics (SPEC-3 & SPEC-4)
- **Individual Wage Tracking**: Agent-based wage modeling with job-to-job transition mechanics
- **Nationality Segmentation**: Tracks worker nationalities with realistic H-1B distributions
- **Job Mobility**: Temporary workers 20% less likely to change jobs (based on Jennifer Hunt research)
- **Live Data Fetching**: Can fetch current statistics from authoritative sources with `--live-fetch`

### Policy Analysis (SPEC-5)
- **Per-Country Cap System**: Optional 7% per-country limit on employment-based green cards
- **FIFO Queue Management**: Separate nationality queues with backlog tracking
- **Comparative Analysis**: Run simulations with and without per-country caps

### Enhanced Visualization (SPEC-6)
- **Interactive Charts**: Comparative wage analysis with publication-quality visualizations
- **Scenario Comparison**: Automatic side-by-side comparison of capped vs uncapped scenarios
- **Multiple Output Formats**: PNG for reports, HTML for interactive analysis
- **Comprehensive Dashboard**: All-in-one view of simulation results

### **NEW: Backlog Analysis by Nationality (SPEC-7)**
- **Comparative Backlog Tracking**: Compare final-year backlogs by nationality between policy scenarios
- **CSV Export**: Detailed backlog data for all nationalities including zero-backlog countries
- **Backlog Visualizations**: Bar charts and interactive plots showing policy impact on different nationalities
- **Integrated Analysis**: Seamlessly combines with existing wage and workforce visualizations

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository:**
