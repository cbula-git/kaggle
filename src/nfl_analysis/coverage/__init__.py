"""
Coverage Analysis Module
========================
Advanced defensive coverage analysis including zone responsibilities,
route synergy, and offensive advantage calculations.

Note: Visualization classes (CoverageVisualizer, CoveragePlotHelper) have been
moved to the nfl_analysis.visualization module.
"""

from nfl_analysis.coverage.coverage_area_analyzer import CoverageAreaAnalyzer
from nfl_analysis.coverage.zone_coverage import ZoneCoverage

__all__ = [
    "CoverageAreaAnalyzer",
    "ZoneCoverage",
]
