"""Main script for ocean plastics data processing."""

import os
from pathlib import Path
from src.data.processor import process_doer_data, process_taiwan_data, integrate_datasets
from src.data.visualizer import generate_summary_visualizations


def main():
    """Main function to process all datasets and generate visualizations."""
    # Check if data directory exists, otherwise use current directory
    data_dir = 'data' if os.path.exists('data') else '.'

    # Define file paths with flexible directory handling
    doer_file = os.path.join(data_dir, 'DOER  Microplastics Database.csv')

    # Check if the file exists, try alternative naming
    if not os.path.exists(doer_file):
        alternative_names = [
            os.path.join(data_dir, 'DOER_Microplastics_Database.csv'),
            os.path.join(data_dir, 'DOER_microplastics_database.csv'),
            os.path.join(data_dir, 'DOER Microplastics Database.csv')
        ]

        for alt_name in alternative_names:
            if os.path.exists(alt_name):
                doer_file = alt_name
                print(f"Found DOER file at: {doer_file}")
                break

    particle_files = [
        os.path.join(data_dir, 'Longmen Beach particle count.csv'),
        os.path.join(data_dir, 'Xialiao Beach particle count.csv')
    ]

    shape_files = [
        os.path.join(data_dir, 'Longmen Beach shape count.csv'),
        os.path.join(data_dir, 'Xialiao Beach shape count.csv')
    ]

    color_files = [
        os.path.join(data_dir, 'Longmen Beach color count.csv'),
        os.path.join(data_dir, 'Xialiao Beach color count.csv')
    ]

    # Create output directory
    output_dir = 'outputs'

    # Process datasets
    doer_df = process_doer_data(doer_file)
    taiwan_df = process_taiwan_data(particle_files, shape_files, color_files)

    # Integrate datasets
    integrated_df = integrate_datasets(doer_df, taiwan_df)

    # Save processed data
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    integrated_df.to_csv(f'{output_dir}/integrated_ocean_plastics.csv', index=False)

    # Generate visualizations
    generate_summary_visualizations(integrated_df, output_dir)

    print("Data processing complete!")


if __name__ == "__main__":
    main()