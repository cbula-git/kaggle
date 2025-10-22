"""
NFL Analysis Package
====================
A Python package for analyzing NFL tracking data from Kaggle competitions.

This package provides tools for:
- Data consolidation from weekly CSV files
- Data loading and querying
- Exploratory data analysis
- Play animation and visualization
- Feature engineering utilities
- Coverage analysis and defensive metrics
"""

__version__ = "0.1.0"
__author__ = "NFL Analysis Team"

# Import main classes for convenient access
from nfl_analysis.consolidation.consolidator import NFLDataConsolidator
from nfl_analysis.io.loader import NFLDataLoader
from nfl_analysis.exploration.explorer import NFLDataExplorer
from nfl_analysis.coverage.coverage_area_analyzer import CoverageAreaAnalyzer
from nfl_analysis.coverage.zone_coverage import ZoneCoverage

__all__ = [
    "NFLDataConsolidator",
    "NFLDataLoader",
    "NFLDataExplorer",
    "CoverageAreaAnalyzer",
    "ZoneCoverage",
]
