# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NFL tracking data analysis project for a Kaggle competition, organized as a Python package (`nfl_analysis`). The package consolidates weekly NFL tracking data files into unified datasets and provides tools for analyzing player movements, routes, and spatial relationships during passing plays.

## Package Structure

```
kaggle/
├── src/nfl_analysis/              # Main package
│   ├── consolidation/             # Data consolidation
│   │   └── consolidator.py        # NFLDataConsolidator class
│   ├── io/                        # Data loading
│   │   └── loader.py              # NFLDataLoader class
│   ├── exploration/               # Data exploration
│   │   └── explorer.py            # NFLDataExplorer class
│   ├── coverage/                  # Coverage analysis
│   │   ├── coverage_area_analyzer.py  # Zone stress, route synergy
│   │   └── zone_coverage.py       # Cover 2/3 definitions
│   ├── visualization/             # Play visualization and plotting
│   │   ├── animator.py            # PlayAnimator class
│   │   ├── field_renderer.py     # FieldRenderer for drawing fields
│   │   ├── coverage_visualizer.py # Coverage analysis plots
│   │   ├── route_visualizer.py    # Route analysis plots
│   │   ├── plot_utils.py          # Plotting helper utilities
│   │   └── config.py              # Unified styling and configuration
│   └── utils/                     # Shared utilities
│       └── metrics.py             # Distance, separation, etc.
├── scripts/                       # CLI entry points
│   ├── consolidate.py             # Data consolidation CLI
│   ├── explore.py                 # Data exploration CLI
│   └── animate.py                 # Animation CLI
├── tests/                         # Test suite
│   ├── conftest.py                # Pytest fixtures
│   ├── test_metrics.py            # Metrics tests
│   └── test_loader.py             # Loader tests
├── notebooks/                     # Jupyter analysis notebooks
│   ├── derived_metrics.ipynb      # Feature engineering
│   ├── attention.ipynb            # Attention mechanisms
│   ├── geometric_coordination.ipynb
│   └── animation.py               # Play animation script
├── data/                          # All data files
│   ├── raw/                       # Raw Kaggle data
│   └── consolidated/              # Consolidated outputs
├── requirements.txt               # Package dependencies
├── setup.py                       # Package installation
├── CLAUDE.md                      # This file
└── MIGRATION.md                   # Migration guide
```

## Installation and Setup

### Package Installation

```bash
# Install package in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"

# Install just the requirements
pip install -r requirements.txt
```

### Environment Setup

```bash
# Activate virtual environment (if exists)
source .venv/bin/activate
```

## Common Commands

### Data Consolidation

```bash
# Using CLI script
python scripts/consolidate.py

# With custom paths
python scripts/consolidate.py --data-dir data/raw/... --output-dir data/consolidated

# Skip spatial features (faster)
python scripts/consolidate.py --skip-spatial

# Using package directly in Python
from nfl_analysis import NFLDataConsolidator
consolidator = NFLDataConsolidator('data/raw/...', 'data/consolidated')
consolidator.consolidate_all(skip_spatial=False)
```

### Data Exploration

```bash
# Using CLI script
python scripts/explore.py

# Skip visualizations
python scripts/explore.py --skip-viz

# Using package directly
from nfl_analysis import NFLDataExplorer
explorer = NFLDataExplorer('data/consolidated')
explorer.run_full_exploration()
```

### Data Loading

```python
from nfl_analysis import NFLDataLoader

# Initialize loader
loader = NFLDataLoader('data/consolidated')

# Load specific datasets
input_df = loader.load_input()
play_df = loader.load_play_level()

# Get data for specific play
play_data = loader.get_play_data(game_id=2023090700, play_id=1679)

# Get data for specific player
player_data = loader.get_player_data(nfl_id=46243, limit=10)

# Get week data
week_data = loader.get_week_data(week=1)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/nfl_analysis --cov-report=html

# Run specific test file
pytest tests/test_metrics.py

# Run specific test
pytest tests/test_metrics.py::TestCalculateDistance::test_zero_distance
```

### Working with Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Notebooks are in notebooks/ directory
# Key notebooks:
# - derived_metrics.ipynb - Feature engineering for offensive separation
# - attention.ipynb - Attention-based metrics
# - geometric_coordination.ipynb - Spatial relationship analysis
# - gravity.ipynb - Gravitational influence models
# - route_tree.ipynb - Route classification and analysis
```

### Play Animation

```bash
# Using CLI script (recommended)
python scripts/animate.py <game_id> <play_id>

# Save to file
python scripts/animate.py 2023090700 1679 --save play.mp4

# Save as GIF with custom settings
python scripts/animate.py 2023090700 1679 --save play.gif --fps 5 --interval 150

# Custom pause time when displaying
python scripts/animate.py 2023090700 1679 --pause 60

# Using notebook script (alternative)
python notebooks/animation.py <game_id> <play_id>

# Example:
python notebooks/animation.py 2023090700 1679

# Using package directly in Python
from nfl_analysis import PlayAnimator

animator = PlayAnimator(data_dir='data/consolidated')
ani = animator.create_animation(game_id=2023090700, play_id=1679)
animator.show(ani, pause_time=30)

# Or save it
animator.save(ani, 'my_play.mp4', fps=10)
```

## Architecture and Design Patterns

### Data Pipeline Flow

1. **Raw Data** (`data/raw/114239_nfl_competition_files_published_analytics_final/train/`)
   - Weekly input files: `input_2023_w*.csv` (pre-pass tracking)
   - Weekly output files: `output_2023_w*.csv` (post-pass tracking)
   - Supplementary: `supplementary_data.csv` (play context)

2. **Consolidation** (`nfl_analysis.consolidation`)
   - `NFLDataConsolidator` combines weekly files
   - Adds `week` column to all records
   - Creates derived datasets
   - Saves to `data/consolidated/` as Parquet files

3. **Consolidated Datasets** (`data/consolidated/*.parquet`)
   - `master_input.parquet` - All pre-pass tracking frames
   - `master_output.parquet` - All post-pass tracking frames
   - `supplementary.parquet` - Play context
   - `play_level.parquet` - Play-aggregated statistics
   - `trajectories.parquet` - Complete player paths
   - `player_analysis.parquet` - Player-centric features
   - `route_analysis.parquet` - Receiver route metrics and success rates
   - `route_trajectories.parquet` - Normalized route trajectories for visualization
   - `spatial_features.parquet` - Relative positions/distances

4. **Data Access** (`nfl_analysis.io`)
   - `NFLDataLoader` provides convenient access methods
   - Supports filtering by play, player, week
   - Provides dataset metadata

5. **Analysis** (notebooks/)
   - Jupyter notebooks for exploratory analysis
   - Feature engineering experiments
   - Visualization and animation

### Key Design Patterns

**Data Keys**: All datasets share common identifier columns:
- Play level: `['game_id', 'play_id']`
- Player level: `['game_id', 'play_id', 'nfl_id']`
- Frame level: `['game_id', 'play_id', 'nfl_id', 'frame_id']`

**Phase Tracking**: Trajectories dataset uses `phase` column:
- `'pre_pass'` - Input tracking before pass release
- `'post_pass'` - Output tracking after pass release
- Frame IDs renumbered to be continuous across phases

**Player Roles**: Key column for filtering:
- `'Passer'` - Quarterback
- `'Targeted Receiver'` - Intended receiver
- `'Other Route Runner'` - Other offensive receivers
- `'Defensive Coverage'` - Defenders covering receivers

## Important Implementation Notes

### Spatial Features Performance
The `create_spatial_features()` method is computationally expensive:
- Iterates over every frame of every play
- Calculates pairwise distances between players
- Can take significant time on large datasets

Use `skip_spatial=True` during development.

### Frame ID Continuity
When working with trajectories:
- Input frames: original `frame_id` values (typically 1 to ~15-25)
- Output frames: `frame_id` adjusted by adding `max_input_frame`
- Creates continuous timeline across the pass event

### Player Prediction Targets
The `player_to_predict` boolean indicates which players need trajectory predictions. The `player_analysis` dataset focuses only on these players.

### Missing Data Handling
- Spatial features may have `None` for `dist_to_target` if no targeted receiver exists
- Player analysis uses left joins with supplementary data
- Output data may not exist for all players in input data

## Coordinate System

- **X-axis**: 0 to 120 yards (includes 10-yard end zones)
- **Y-axis**: 0 to 53.3 yards (field width)
- Ball landing position: `ball_land_x`, `ball_land_y`
- Field visualized with green background and white yard markers

## Utility Functions

The `nfl_analysis.utils.metrics` module provides:
- `calculate_distance(x1, y1, x2, y2)` - Euclidean distance
- `calculate_separation(off_pos, def_pos)` - Player separation
- `calculate_velocity(x1, y1, x2, y2, time_delta)` - Velocity calculation
- `calculate_displacement(trajectory_df)` - Total displacement
- `calculate_path_length(trajectory_df)` - Total path length
- `calculate_angle(x1, y1, x2, y2)` - Angle calculation
- `calculate_relative_position(...)` - Relative position metrics
- `calculate_acceleration(v1, v2, time_delta)` - Acceleration

## Analysis Patterns

### Calculating Player Separation
See `notebooks/derived_metrics.ipynb` for examples of:
- Separation between offensive/defensive players at throw time
- Separation at ball landing time
- Min/max/mean separation by player and game

### Visualization and Animation Workflow
The visualization module (`nfl_analysis.visualization`) provides:
- `PlayAnimator` class for creating play animations
- `FieldRenderer` class for drawing NFL fields with proper dimensions
- `CoverageVisualizer` class for coverage analysis plots
- `RouteVisualizer` class for route analysis and receiver performance
- `CoveragePlotHelper` for reusable plotting utilities
- Automatic loading and merging of play data
- Color-coded player roles and trajectories
- Unified configuration via `visualization.config`

**Animation Features:**
- Displays complete player paths (pre-pass and post-pass)
- Shows ball landing position
- Includes play description and game context
- Supports saving to video (MP4, GIF, etc.)
- Customizable frame rate and timing

**Example Usage:**
```python
from nfl_analysis import PlayAnimator, FieldRenderer

# Basic usage
animator = PlayAnimator(data_dir='data/consolidated')
ani = animator.create_animation(game_id=2023090700, play_id=1679)
animator.show(ani)

# Custom field size
custom_renderer = FieldRenderer(figsize=(16, 8))
animator = PlayAnimator(data_dir='data/consolidated', field_renderer=custom_renderer)

# Create and save
ani = animator.create_animation(game_id=2023090700, play_id=1679, interval=100)
animator.save(ani, 'amazing_play.gif', fps=10)
```

### Route Analysis and Visualization
The `RouteVisualizer` class provides comprehensive route analysis tools:

**Features:**
- Route frequency distribution
- Success rate (completion %) by route type
- Average separation (distance to ball at landing)
- Combined dashboard view
- Player-specific filtering

**Example Usage:**
```python
from nfl_analysis import RouteVisualizer

# Initialize visualizer
visualizer = RouteVisualizer(data_dir='data/consolidated')

# Load route data
route_data = visualizer.load_route_data()

# Plot route frequency for all players
fig, ax = visualizer.plot_route_frequency()

# Plot success rate by route type
fig, ax = visualizer.plot_route_success_rate(min_attempts=10)

# Plot average separation by route type
fig, ax = visualizer.plot_route_separation(min_attempts=10)

# Create comprehensive dashboard
fig, axes = visualizer.plot_route_dashboard(min_attempts=10)

# Filter by specific player
fig, ax = visualizer.plot_route_frequency(player_name="Tyreek")
fig, ax = visualizer.plot_route_success_rate(nfl_id=47851)
```

**Route Analysis Dataset:**
The `route_analysis.parquet` dataset contains one row per targeted receiver per play with:
- Route type (HITCH, OUT, FLAT, CROSS, GO, IN, SLANT, POST, ANGLE, CORNER, SCREEN, WHEEL)
- Position at pass release (x, y, speed, acceleration, direction)
- Position at ball landing (x, y)
- Distance to ball at landing (`dist_to_ball_at_landing`)
- Route metrics (depth, width, distance) - **normalized by play direction**
- Success indicator (`is_completion`)
- Coverage context (coverage type, formation, alignment)

**Route Trajectories Dataset:**
The `route_trajectories.parquet` dataset contains normalized frame-by-frame route trajectories for visualization:
- Normalized coordinates (`x_norm`, `y_norm`) - all routes start at origin (0,0)
- Original coordinates preserved (`x_original`, `y_original`)
- Play direction standardized - all routes normalized to go "right"
- Frame sequence within each route (`frame_sequence`)
- Distance traveled from start (`dist_from_start`)
- Phase indicator (`pre_pass` or `post_pass`)
- Route metadata (type, completion, coverage)

**Route Overlay Visualization:**
Visualize all routes for a player in a game with color-coded completion status:

```python
from nfl_analysis.visualization import RouteVisualizer

visualizer = RouteVisualizer(data_dir='data/consolidated')

# Plot all routes for a player in a specific game
fig, ax = visualizer.plot_player_routes_overlay(
    game_id=2023123105,
    player_name='Davante Adams',
    save_path='davante_routes.png'
)

# Use NFL ID for exact matching
fig, ax = visualizer.plot_player_routes_overlay(
    game_id=2023091709,
    nfl_id=56042,  # Puka Nacua
    figsize=(18, 12)
)
```

Features:
- **Green routes** = Completions
- **Red routes** = Incompletions
- **Blue star** = Starting position (all routes normalized to 0,0)
- Route type labels at endpoints
- Completion percentage in title

### Feature Engineering
Notebooks explore various features:
- Distance metrics (Euclidean distance between players)
- Velocity and acceleration patterns
- Route shape classification
- Spatial influence models (gravity)
- Attention-weighted coverage metrics

## Development Workflow

1. **Making Changes**: Always work on a feature branch
2. **Testing**: Run tests before committing (`pytest`)
3. **Package Updates**: If modifying package structure, reinstall (`pip install -e .`)
4. **Data Changes**: If data structure changes, update both consolidator and loader

## Migration from Legacy Structure

If transitioning from the old structure:
- Old `consolidate_nfl_data.py` → `src/nfl_analysis/consolidation/consolidator.py`
- Old `nfl_data_loader.py` → `src/nfl_analysis/io/loader.py`
- Old `explore_consolidated_data.py` → `src/nfl_analysis/exploration/explorer.py`
- Old `notebook/` → `notebooks/`
- Old `consolidated_data/` → `data/consolidated/`
- Old raw data path → `data/raw/114239_nfl_competition_files_published_analytics_final/`

Use the new package imports:
```python
from nfl_analysis import NFLDataConsolidator, NFLDataLoader, NFLDataExplorer
from nfl_analysis import PlayAnimator, FieldRenderer, CoverageVisualizer, RouteVisualizer
from nfl_analysis.utils.metrics import calculate_distance, calculate_separation
from nfl_analysis.visualization import CoveragePlotHelper
```
