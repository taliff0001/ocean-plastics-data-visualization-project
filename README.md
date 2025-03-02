# Ocean Plastics Analysis Project

A comprehensive data analysis and visualization project for studying microplastic distribution in marine environments, with a special focus on Taiwan beach zones.

## Project Description

This project provides data processing and visualization tools for analyzing microplastic data from multiple sources:
1. The DOER Microplastics Database (global data)
2. Field samples from Longmen and Xialiao beaches in Taiwan

The package produces visualizations showing distribution patterns, shape composition, concentration levels, and temporal trends of microplastics across different marine environments.

## Installation

### Setting Up the Conda Environment

```bash
# Create a new conda environment
conda create -n ocean-plastics python=3.9

# Activate the environment
conda activate ocean-plastics

# Install core dependencies
conda install numpy pandas matplotlib seaborn

# Install development tools
conda install -c conda-forge flake8 black mypy

# Alternatively, use pip to install from requirements.txt
pip install -r requirements.txt
```

### Verify the Installation

```bash
# Check that packages are installed
python -c "import numpy; import pandas; import matplotlib; import seaborn; print('All core packages imported successfully')"

# Check development tools
flake8 --version
black --version
mypy --version
```

## Usage

The project is now organized as a package with a central command-line interface:

```bash
# Process and integrate datasets
python main.py process

# Generate beach zonation visualization
python main.py visualize beach

# Generate summary visualizations
python main.py visualize summary

# Get help
python main.py --help
```

## Development Commands

- Lint: `flake8 src/`
- Format: `black src/`
- Type check: `mypy src/`

## Code Style Guidelines

- **Naming**: snake_case for variables/functions; CamelCase for classes
- **Imports**: 1) standard library, 2) third-party packages, 3) local modules
- **Documentation**: Use docstrings with Args/Returns sections
- **Formatting**: 4-space indentation, 79 character line limit (PEP 8)
- **Error handling**: Prefer conditional checks over try/except where appropriate
- **File organization**: Package-based organization with separate modules for different functionality
- **Data processing**: Raw data in data/ directory; processed outputs in outputs/

## Project Structure

```
ocean-plastics/
├── data/                       # Raw data files
├── outputs/                    # Generated visualizations and processed data
├── src/                        # Package source code
│   ├── data/                   # Data processing modules
│   │   ├── processor.py        # Core data processing functions
│   │   ├── visualizer.py       # Basic visualization utilities
│   │   └── main.py             # Main data processing script
│   ├── utils/                  # Utility functions
│   └── visualizations/         # Visualization modules
│       ├── beach_zonation.py   # Beach zonation visualization
│       └── [other vis modules] # Additional visualization modules
├── main.py                     # CLI entrypoint
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Requirements

See `requirements.txt` for the full list of dependencies:

```
# Core data processing and visualization
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Development tools
flake8>=4.0.0
black>=22.0.0
mypy>=0.910
```

## Data Sources

- DOER Microplastics Database: A collection of global microplastics data
- Taiwan Beach Samples: Field samples collected from Longmen and Xialiao beaches between 2018-2019
