# NFL Tracking Data Consolidation

This package consolidates NFL tracking data from multiple weekly CSV files into unified datasets for comprehensive analysis.

## Scripts

### 1. `consolidate_nfl_data.py`
Main consolidation script that combines all weekly data files.

**Usage:**
bash python consolidate_nfl_data.py

**Output:**
- `consolidated_data/master_input.parquet` - All pre-pass tracking data
- `consolidated_data/master_output.parquet` - All post-pass tracking data
- `consolidated_data/supplementary.parquet` - Play context information
- `consolidated_data/play_level.parquet` - Play-level aggregated features
- `consolidated_data/trajectories.parquet` - Complete player trajectories
- `consolidated_data/player_analysis.parquet` - Player-centric analysis data
- `consolidated_data/spatial_features.parquet` - Spatial relationship features
- `consolidated_data/consolidation_summary.txt` - Summary report

**Options:**
To skip spatial feature calculation (saves time):

```
python consolidator = NFLDataConsolidator(DATA_DIR, OUTPUT_DIR) consolidator.consolidate_all(skip_spatial=True)
```
### 2. `explore_consolidated_data.py`
Exploration and validation script for consolidated data.

**Usage:**
`bash python explore_consolidated_data.py`

**Output:**
- Data quality checks
- Summary statistics
- Visualizations in `visualizations/` directory
- Exploration report

### 3. `nfl_data_loader.py`
Utility module for easy data loading in your analyses.

**Usage:**
python from nfl_data_loader import NFLDataLoader
# Initialize loader
loader = NFLDataLoader()
# Load specific datasets
input_df = loader.load_input() play_df = loader.load_play_level() player_df = loader.load_player_analysis()
# Get data for specific play
play_data = loader.get_play_data(game_id=2023090700, play_id=56)
# Get data for specific player
player_data = loader.get_player_data(nfl_id=12345, limit=10)
# Load all datasets
all_data = loader.load_all()


## Quick Start

1. **Consolidate the data:**
   ```bash
   python consolidate_nfl_data.py
   ```
   
2. **Explore the consolidated data:**
   ```bash
   python explore_consolidated_data.py
   ```
   
3. **Use in your analysis:**
   ```python
   from nfl_data_loader import load_play_level, load_player_analysis
   
   play_df = load_play_level()
   player_df = load_player_analysis()
   
   # Your analysis here...
   ```

## Data Structure

### Master Input (`master_input.parquet`)
Pre-pass tracking data with player positions, velocities, and context.

**Key Columns:**
- `game_id`, `play_id`, `nfl_id`, `frame_id`
- `x`, `y` - Player position
- `s`, `a` - Speed and acceleration
- `o`, `dir` - Orientation and direction
- `player_position`, `player_role`, `player_side`
- `ball_land_x`, `ball_land_y` - Ball landing position
- `week` - Week number (added during consolidation)

### Master Output (`master_output.parquet`)
Post-pass tracking data with player positions after the pass.

**Columns:**
- `game_id`, `play_id`, `nfl_id`, `frame_id`
- `x`, `y` - Player position
- `week` - Week number

### Play Level (`play_level.parquet`)
Aggregated statistics and context for each play.

**Key Features:**
- Aggregate statistics (mean, std, max) for speed, acceleration, positions
- Play context from supplementary data
- Number of frames and players
- Ball landing position

### Player Analysis (`player_analysis.parquet`)
Player-centric features focusing on prediction targets.

**Key Features:**
- Last pre-pass position and velocity
- First and last post-pass positions
- Displacement calculations
- Distance to ball landing
- Play context

### Trajectories (`trajectories.parquet`)
Complete player movement sequences (pre + post pass).

**Key Features:**
- Combined input and output data
- Continuous frame_id numbering
- Phase indicator (pre_pass/post_pass)

### Spatial Features (`spatial_features.parquet`)
Relative positions and distances between players.

**Key Features:**
- Distance to passer
- Distance to ball landing
- Distance to targeted receiver
- Relative positions and angles

## Requirements
pandas numpy pyarrow # for parquet support matplotlib seaborn

Install with:
```bash
pip install pandas numpy pyarrow matplotlib seaborn
```