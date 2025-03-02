# Project Guidelines for AI Assistants

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
- Visualization scripts (beach-zonation, plastic-composition, temporal-trends, global-comparison)
- Data processing code in data-processing-code.py
- Raw data in CSV format
- Outputs saved as PNG files and processed CSV