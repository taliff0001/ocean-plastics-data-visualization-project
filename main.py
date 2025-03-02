"""
Ocean Plastics Analysis - Main Script
This script serves as the entry point for the ocean plastics analysis project,
providing access to both data processing and visualization functionality.
"""

import argparse
from pathlib import Path


def setup_argparse():
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ocean Plastics Analysis Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Data processing command
    process_parser = subparsers.add_parser("process", help="Process and integrate datasets")
    
    # Visualization commands
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_subparsers = viz_parser.add_subparsers(dest="viz_type", help="Visualization type")
    
    # Beach zonation visualization
    beach_parser = viz_subparsers.add_parser("beach", help="Beach zonation visualization")
    
    # Temporal trends visualization
    temporal_parser = viz_subparsers.add_parser("temporal", help="Temporal trends visualization")
    
    # Plastic composition visualization
    composition_parser = viz_subparsers.add_parser("composition", help="Plastic composition visualization")
    
    # Global comparison visualization
    global_parser = viz_subparsers.add_parser("global", help="Global comparison visualization")
    
    # Summary (all) visualizations
    summary_parser = viz_subparsers.add_parser("summary", help="Generate all summary visualizations")
    
    return parser


def process_data():
    """Process and integrate the datasets."""
    from src.data.main import main as data_main
    data_main()
    

def visualize_beach_zonation():
    """Generate beach zonation visualization."""
    from src.visualizations.beach_zonation import main as beach_main
    beach_main()


def visualize_summary():
    """Generate all summary visualizations."""
    # This uses the existing processed data
    from src.data.visualizer import generate_summary_visualizations
    import pandas as pd
    
    # Check if processed data exists
    integrated_data_path = Path('outputs/integrated_ocean_plastics.csv')
    if not integrated_data_path.exists():
        print("Processed data not found. Running data processing first...")
        process_data()
    
    # Load integrated data
    integrated_df = pd.read_csv(integrated_data_path)
    
    # Generate visualizations
    generate_summary_visualizations(integrated_df, 'outputs')


def main():
    """Main entry point for the application."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # If no command is provided, print help
    if not args.command:
        parser.print_help()
        return
    
    # Execute requested command
    if args.command == "process":
        process_data()
    
    elif args.command == "visualize":
        if not args.viz_type:
            parser.parse_args(["visualize", "-h"])
            return
            
        if args.viz_type == "beach":
            visualize_beach_zonation()
        elif args.viz_type == "summary":
            visualize_summary()
        # Other visualization types would be implemented here
        else:
            print(f"Visualization type '{args.viz_type}' not yet implemented")
    
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()