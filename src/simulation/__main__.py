"""
Entry point for running the simulation as a module.
Allows execution with: python -m src.simulation
"""

from .cli import main

if __name__ == "__main__":
    exit(main())
