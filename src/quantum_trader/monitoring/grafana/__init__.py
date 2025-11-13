"""Grafana integration module for quantum trader monitoring.

This module provides integration with Grafana for creating and managing dashboards
that visualize trading metrics, system performance, and business analytics.

Features:
    - Dashboard creation and management
    - Panel configuration (graphs, gauges, tables, heatmaps)
    - Data source integration with Prometheus
    - Alert rule configuration in Grafana
    - Template variable management
    - Dashboard templating and reusability
    - Dynamic dashboard generation
    - Annotation management
    - Dashboard versioning and history
    - Multi-workspace dashboard organization

Supported Visualizations:
    - Time series charts
    - Gauges and stat panels
    - Tables and data grids
    - Heatmaps and histograms
    - Pie charts and donut charts
    - Map visualizations
    - Bar and column charts
    - Scatter plots and bubble charts
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add Grafana integration classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
