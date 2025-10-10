# tests/test_backlog_analysis.py
"""
Unit tests for backlog analysis functionality.
Tests backlog tracking, CSV export, and visualization integration.
NEW FOR SPEC-7: Comparative backlog analysis by nationality.
"""

import pytest
import tempfile
import os
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from collections import deque

from src.simulation.models import (
    SimulationConfig, BacklogAnalysis, TemporaryWorker, Worker, WorkerStatus
)
from src.simulation.sim import Simulation
from src.simulation.utils import save_backlog_analysis, load_backlog_analysis
from src.simulation.empirical_params import TEMP_NATIONALITY_DISTRIBUTION

# Test if visualization modules are available
try:
    from src.simulation.visualization import (
        SimulationVisualizer, validate_backlog_dataframes
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

class TestBacklogAnalysis:
    """Test cases for backlog analysis functionality."""
    
    def test_backlog_analysis_creation_uncapped(self):
        """Test BacklogAnalysis creation for uncapped scenario."""
        # Create mock simulation with global queue
        mock_sim = MagicMock()
        mock_sim.country_cap_enabled = False
        mock_sim.country_queues = {}
        
        # Create mock global queue with nationality-distributed workers
        mock_sim.global_queue = deque([
            TemporaryWorker(1, 2025, "India"),
            TemporaryWorker(2, 2025, "China"), 
            TemporaryWorker(3, 2025, "India"),
            TemporaryWorker(4, 2025, "Canada")
        ])
        
        # Mock states
        mock_sim.states = [MagicMock()]
        mock_sim.states[0].year = 2030
        
        # Create backlog analysis
        analysis = BacklogAnalysis.from_simulation(mock_sim, "uncapped")
        
        assert analysis.scenario == "uncapped"
        assert analysis.final_year == 2030
        assert analysis.total_backlog == 4
        assert analysis.backlog_by_nationality["India"] == 2
        assert analysis.backlog_by_nationality["China"] == 1
        assert analysis.backlog_by_nationality["Canada"] == 1
        # Other nationalities should have 0
        assert analysis.backlog_by_nationality["Philippines"] == 0
    
    def test_backlog_analysis_creation_capped(self):
        """Test BacklogAnalysis creation for capped scenario."""
        # Create mock simulation with country queues
        mock_sim = MagicMock()
        mock_sim.country_cap_enabled = True
        mock_sim.global_queue = None
        
        # Create mock country queues
        mock_sim.country_queues = {
            "India": deque([TemporaryWorker(1, 2025, "India"), TemporaryWorker(2, 2025, "India")]),
            "China": deque([TemporaryWorker(3, 2025, "China")]),
            "Canada": deque()  # Empty queue
        }
        
        # Mock states
        mock_sim.states = [MagicMock()]
        mock_sim.states[0].year = 2030
        
        # Create backlog analysis
        analysis = BacklogAnalysis.from_simulation(mock_sim, "capped")
        
        assert analysis.scenario == "capped"
        assert analysis.final_year == 2030
        assert analysis.total_backlog == 3
        assert analysis.backlog_by_nationality["India"] == 2
        assert analysis.backlog_by_nationality["China"] == 1
        assert analysis.backlog_by_nationality["Canada"] == 0
    
    def test_backlog_analysis_to_dataframe(self):
        """Test conversion of BacklogAnalysis to DataFrame."""
        # Create test backlog analysis
        backlog_data = {nationality: 0 for nationality in TEMP_NATIONALITY_DISTRIBUTION.keys()}
        backlog_data["India"] = 100
        backlog_data["China"] = 50
        
        analysis = BacklogAnalysis(
            scenario="capped",
            backlog_by_nationality=backlog_data,
            total_backlog=150,
            final_year=2030
        )
        
        df = analysis.to_dataframe()
        
        # Check DataFrame structure
        assert 'nationality' in df.columns
        assert 'backlog_size' in df.columns
        assert 'scenario' in df.columns
        
        # Check data content
        assert len(df) == len(TEMP_NATIONALITY_DISTRIBUTION)
        assert all(df['scenario'] == 'capped')
        
        # Check specific values
        india_row = df[df['nationality'] == 'India'].iloc[0]
        assert india_row['backlog_size'] == 100
        
        china_row = df[df['nationality'] == 'China'].iloc[0]
        assert china_row['backlog_size'] == 50
    
    def test_backlog_analysis_get_top_backlogs(self):
        """Test getting top backlogs by nationality."""
        backlog_data = {
            "India": 1000,
            "China": 500,
            "Canada": 100,
            "Philippines": 50,
            "South Korea": 25,
            "United Kingdom": 10,
            "Mexico": 5,
            "Brazil": 2,
            "Germany": 1,
            "Other": 0
        }
        
        analysis = BacklogAnalysis(
            scenario="test",
            backlog_by
