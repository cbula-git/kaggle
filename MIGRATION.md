# Migration Guide: Legacy to Package Structure

This guide helps you migrate from the old flat file structure to the new `nfl_analysis` package structure.

## What Changed?

### Directory Structure

**Before:**
```
kaggle/
├── consolidate_nfl_data.py
├── nfl_data_loader.py
├── explore_consolidated_data.py
├── notebook/
│   └── animation.py
├── consolidated_data/
└── 114239_nfl_competition_files_published_analytics_final/
```

**After:**
```
kaggle/
├── src/nfl_analysis/           # Python package
├── scripts/                    # CLI tools
├── tests/                      # Test suite
├── notebooks/                  # Jupyter notebooks (renamed)
├── data/
│   ├── raw/                    # Raw data
│   └── consolidated/           # Consolidated data
├── requirements.txt
└── setup.py
```

### Code Organization

| Old Location | New Location |
|-------------|--------------|
| `consolidate_nfl_data.py` | `src/nfl_analysis/consolidation/consolidator.py` |
| `nfl_data_loader.py` | `src/nfl_analysis/io/loader.py` |
| `explore_consolidated_data.py` | `src/nfl_analysis/exploration/explorer.py` |
| `notebook/` | `notebooks/` |
| `consolidated_data/` | `data/consolidated/` |
| Raw data directory | `data/raw/114239...` |

## Migration Steps

### 1. Install the Package

```bash
# Install in development mode
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

### 2. Update Your Scripts

#### Old Way:
```python
# Direct script execution
python consolidate_nfl_data.py
python explore_consolidated_data.py
```

#### New Way:
```python
# Using CLI scripts
python scripts/consolidate.py
python scripts/explore.py

# Or after installation (if entry points work):
# nfl-consolidate
# nfl-explore
```

### 3. Update Your Imports

#### Old Way:
```python
from nfl_data_loader import NFLDataLoader, load_input, load_play_level

loader = NFLDataLoader('consolidated_data')
df = load_input('consolidated_data')
```

#### New Way:
```python
from nfl_analysis import NFLDataLoader
from nfl_analysis.io import load_input, load_play_level

loader = NFLDataLoader('data/consolidated')
df = load_input('data/consolidated')
```

### 4. Update Data Paths

#### In Python Code:
```python
# Old paths
'consolidated_data/master_input.parquet'
'114239_nfl_competition_files_published_analytics_final/train/...'

# New paths
'data/consolidated/master_input.parquet'
'data/raw/114239_nfl_competition_files_published_analytics_final/train/...'
```

#### In Notebooks:
```python
# Old
base_dir = "../consolidated_data/"

# New
base_dir = "../data/consolidated/"
```

### 5. Update Notebooks

Your Jupyter notebooks in `notebooks/` will need path updates:

```python
# Update data loading paths
# Old: base_dir = "../consolidated_data/"
# New: base_dir = "../data/consolidated/"

# You can also use the package now
import sys
sys.path.insert(0, '../src')  # Or just pip install -e .
from nfl_analysis import NFLDataLoader

loader = NFLDataLoader('../data/consolidated')
input_df = loader.load_input()
```

### 6. Run Tests

Verify everything works:

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src/nfl_analysis
```

## Key Differences

### 1. Package Imports

You can now import functionality directly:
```python
from nfl_analysis import (
    NFLDataConsolidator,
    NFLDataLoader,
    NFLDataExplorer
)

from nfl_analysis.utils.metrics import (
    calculate_distance,
    calculate_separation,
    calculate_velocity
)
```

### 2. Default Paths

The default paths have changed:
- Consolidated data: `data/consolidated/` (was `consolidated_data/`)
- Raw data: `data/raw/114239...` (was `114239...`)

### 3. CLI Scripts

New CLI scripts with better argument handling:
```bash
# Consolidation with options
python scripts/consolidate.py --skip-spatial --output-dir data/consolidated

# Exploration with options
python scripts/explore.py --skip-viz --data-dir data/consolidated
```

### 4. Testing

You can now run tests on the package:
```bash
pytest tests/
pytest tests/test_metrics.py
pytest tests/test_loader.py
```

## Backward Compatibility

The old scripts (`consolidate_nfl_data.py`, `nfl_data_loader.py`, `explore_consolidated_data.py`) are still in the repository root for backward compatibility. However, they are deprecated and will be removed in a future version.

To maintain compatibility:
1. The `.gitignore` still ignores both old and new paths
2. Old import paths may work but are not recommended
3. Data can exist in either location temporarily

## Common Issues

### Issue: Module not found
**Solution:** Install the package with `pip install -e .`

### Issue: Data not found
**Solution:** Update paths to use `data/consolidated/` and `data/raw/`

### Issue: Import errors in notebooks
**Solution:** Either:
1. Install package: `pip install -e .` then use `from nfl_analysis import ...`
2. Or add path: `sys.path.insert(0, '../src')` then import

### Issue: Tests failing
**Solution:** Make sure you're in the repository root when running `pytest`

## Recommendations

1. **Start Fresh**: If possible, re-run consolidation with the new scripts
2. **Update Gradually**: Migrate one notebook at a time
3. **Test Thoroughly**: Run tests after each migration step
4. **Use Package Imports**: Take advantage of the new package structure

## Need Help?

- Check `CLAUDE.md` for detailed package documentation
- Look at `tests/` for usage examples
- Review `scripts/` for CLI examples
- Examine `src/nfl_analysis/` for implementation details

## Timeline

- **Phase 1 (Current)**: Both old and new structures coexist
- **Phase 2 (Future)**: Remove old scripts from root
- **Phase 3 (Future)**: Enforce new structure only
