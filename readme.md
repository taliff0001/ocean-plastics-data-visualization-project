# Beach Microplastics Visualization Project

A data visualization project to analyze and present microplastic distribution patterns across beach zones in Taiwan.

## Project Description

This project provides visualization tools for microplastic data collected from Longmen and Xialiao beaches in Taiwan. The visualizations show distribution patterns, shape composition, and concentration levels of microplastics across different beach zones.

## Installation

### Setting Up the Conda Environment

```bash
# Create a new conda environment
conda create -n beach-microplastics python=3.9

# Activate the environment
conda activate beach-microplastics

# Install core dependencies
conda install numpy pandas matplotlib

# Install development tools
conda install -c conda-forge flake8 black mypy

# Alternatively, use pip to install from requirements.txt
pip install -r requirements.txt
```

### Verify the Installation

```bash
# Check that packages are installed
python -c "import numpy; import pandas; import matplotlib; print('All core packages imported successfully')"

# Check development tools
flake8 --version
black --version
mypy --version
```

## Run Commands

- Run scripts: `python <script_name>.py`
- Lint: `flake8 *.py`
- Format: `black *.py`
- Type check: `mypy *.py`

## Code Style Guidelines

- **Naming**: snake_case for variables/functions; CamelCase for classes
- **Imports**: 1) standard library, 2) third-party packages, 3) local modules
- **Documentation**: Use docstrings with Args/Returns sections
- **Formatting**: 4-space indentation, 79 character line limit (PEP 8)
- **Error handling**: Prefer conditional checks over try/except where appropriate
- **File organization**: Separate scripts by visualization type; outputs to outputs/
- **Data processing**: Keep raw data in data/ directory; document in data-dictionary.md

## Project Structure

- **Visualization scripts**:
  - `beach-zonation-visualization.py` - Visualizes microplastic concentration across beach zones
  - `plastic-composition.py` - Analyzes and displays plastic composition data
  - `temporal-trends.py` - Shows changes in microplastic concentration over time
  - `global-comparison.py` - Compares data with other global beach studies
- **Data processing**: `data-processing-code.py`
- **Raw data**: CSV files in `data/` directory
- **Outputs**: PNG visualization files in `outputs/` directory

## Requirements

See `requirements.txt` for the full list of dependencies:

```
# Core data processing and visualization
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0

# Development tools
flake8>=4.0.0
black>=22.0.0
mypy>=0.910
```