# Ocean Plastics Data Processing Script
# This script processes and integrates the DOER Microplastics Database with Taiwan beach datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')


# Function to load and process DOER dataset
def process_doer_data(file_path):
    """
    Process the DOER Microplastics Database.

    Args:
        file_path: Path to the DOER CSV file

    Returns:
        Processed DataFrame
    """
    print(f"Processing DOER data from {file_path}...")

    # Load the data
    doer_df = pd.read_csv(file_path)

    # Filter rows - focus on coastal and marine environments
    env_filter = doer_df['System'].isin(['Marine', 'Estuarine'])
    coastal_filter = doer_df['Zone Area'].str.contains('Coastal|Beach', na=False)
    filtered_df = doer_df[env_filter & coastal_filter].copy()

    print(f"Filtered from {len(doer_df)} to {len(filtered_df)} coastal/marine entries")

    # Handle missing values - avoid chained assignment with inplace=True
    filtered_df = filtered_df.copy()  # Create a true copy to avoid SettingWithCopyWarning
    filtered_df['Subdivison'] = filtered_df['Subdivison'].fillna('Unknown')
    filtered_df['Dominant Size'] = filtered_df['Dominant Size'].fillna('Unknown')
    filtered_df['N'] = pd.to_numeric(filtered_df['N'], errors='coerce').fillna(-1)

    # Standardize text data
    # Normalize shape descriptions
    shape_mapping = {
        'frgmnt': 'fragment',
        'fbr': 'fiber',
        'fbrs': 'fiber',
        'sphr': 'pellet',
        'flm': 'foil',
        'fm': 'foam',
        'flms': 'foil'
    }

    # Function to standardize shapes
    def standardize_shapes(shape_str):
        if pd.isna(shape_str) or shape_str in ['(N/R)', '(N/A)', '(n/r)', '(n/a)']:
            return 'unknown'

        shapes = re.split(r'[,;/\s]+', shape_str.lower())
        standardized = [shape_mapping.get(s, s) for s in shapes if s]
        return ', '.join(sorted(set(standardized)))

    filtered_df['Standardized_Shapes'] = filtered_df['Dominant Shapes'].apply(standardize_shapes)

    # Create year and decade columns for temporal analysis
    # Make sure Sample Time is numeric
    filtered_df['Year'] = pd.to_numeric(filtered_df['Sample Time'], errors='coerce')
    # For decade calculation, handle any remaining NaN values
    filtered_df['Decade'] = ((filtered_df['Year'] // 10) * 10).fillna(-1).astype('Int64')

    # Create log-transformed concentration for visualization
    filtered_df['log_concentration'] = np.log10(filtered_df['mp/kg dw'] + 1)

    # Add source column for later integration
    filtered_df['source'] = 'DOER Database'

    # Create region column for broader geographic analysis
    def assign_region(continent, country):
        if continent == 'Asia':
            if country in ['China', 'Japan', 'South Korea', 'Taiwan']:
                return 'East Asia'
            elif country in ['Thailand', 'Vietnam', 'Malaysia', 'Indonesia', 'Philippines']:
                return 'Southeast Asia'
            else:
                return 'Other Asia'
        else:
            return continent

    filtered_df['region'] = filtered_df.apply(
        lambda row: assign_region(row['Continent'], row['Country']), axis=1
    )

    # Select relevant columns
    selected_columns = [
        'mpID', 'Continent', 'Country', 'region', 'System', 'Waterbody',
        'Zone Area', 'Tidal Zone', 'Test Area', 'Year', 'Decade',
        'mp/kg dw', 'log_concentration', 'MP Unit',
        'Standardized_Shapes', 'Dominant Size', 'Colors',
        'source', 'DOI'
    ]

    return filtered_df[selected_columns]


# Function to process Taiwan beach data
def process_taiwan_data(particle_files, shape_files, color_files):
    """
    Process and integrate Taiwan beach datasets.

    Args:
        particle_files: List of particle count CSV files
        shape_files: List of shape count CSV files
        color_files: List of color count CSV files

    Returns:
        Processed and integrated DataFrame
    """
    print("Processing Taiwan beach data...")

    # Load particle data
    taiwan_dfs = []

    for file_path in particle_files:
        beach_name = Path(file_path).stem.split()[0]
        df = pd.read_csv(file_path)
        df['Beach'] = beach_name
        taiwan_dfs.append(df)

    # Combine particle data
    particle_df = pd.concat(taiwan_dfs, ignore_index=True)

    # Calculate concentration in particles per kg
    # Convert to numeric to ensure proper calculation and handle errors
    particle_df['Particle_count'] = pd.to_numeric(particle_df['Particle_count'], errors='coerce').fillna(0)
    particle_df['Weight_dry_sand_g'] = pd.to_numeric(particle_df['Weight_dry_sand_g'], errors='coerce')

    # Avoid division by zero or missing values
    particle_df['particles_per_kg'] = particle_df.apply(
        lambda row: row['Particle_count'] / (row['Weight_dry_sand_g'] / 1000)
        if row['Weight_dry_sand_g'] > 0 else 0,
        axis=1
    )

    # Create log-transformed concentration for visualization
    particle_df['log_concentration'] = np.log10(particle_df['particles_per_kg'] + 1)

    # Convert date to datetime
    particle_df['Date'] = pd.to_datetime(particle_df['Date_YYYY-MM-DD'])
    particle_df['Year'] = particle_df['Date'].dt.year
    particle_df['Month'] = particle_df['Date'].dt.month
    particle_df['Season'] = pd.cut(
        particle_df['Month'],
        bins=[0, 3, 6, 9, 12],
        labels=['Winter', 'Spring', 'Summer', 'Fall'],
        include_lowest=True
    )

    # Add source column for later integration
    particle_df['source'] = 'Taiwan Beaches'
    particle_df['Continent'] = 'Asia'
    particle_df['Country'] = 'Taiwan'
    particle_df['region'] = 'East Asia'
    particle_df['System'] = 'Marine'
    particle_df['Zone Area'] = 'Beach'

    # Create unique sample ID
    particle_df['sample_id'] = (
            particle_df['Beach'] + '_' +
            particle_df['Date_YYYY-MM-DD'] + '_' +
            particle_df['Transect'] + '_' +
            particle_df['Position'].astype(str) + '_' +
            particle_df['Size_class']
    )

    # Process shape data and merge with particle data
    shape_dfs = []
    for file_path in shape_files:
        beach_name = Path(file_path).stem.split()[0]
        df = pd.read_csv(file_path)
        df['Beach'] = beach_name
        shape_dfs.append(df)

    shape_df = pd.concat(shape_dfs, ignore_index=True)

    # Create unique sample ID for shape data
    shape_df['sample_id'] = (
            shape_df['Beach'] + '_' +
            shape_df['Date_YYYY-MM-DD'] + '_' +
            shape_df['Transect'] + '_' +
            shape_df['Position'].astype(str) + '_' +
            shape_df['Size_class']
    )

    # Process color data and merge with particle data
    color_dfs = []
    for file_path in color_files:
        beach_name = Path(file_path).stem.split()[0]
        df = pd.read_csv(file_path)
        df['Beach'] = beach_name
        color_dfs.append(df)

    color_df = pd.concat(color_dfs, ignore_index=True)

    # Create unique sample ID for color data
    color_df['sample_id'] = (
            color_df['Beach'] + '_' +
            color_df['Date_YYYY-MM-DD'] + '_' +
            color_df['Transect'] + '_' +
            color_df['Position'].astype(str) + '_' +
            color_df['Size_class']
    )

    # Get shape columns and color columns
    shape_columns = [col for col in shape_df.columns if col not in
                     ['Date_YYYY-MM-DD', 'Country_Region', 'Location_name',
                      'Location_lat', 'Location_lon', 'Transect', 'Position',
                      'Size_min_mm', 'Size_max_mm', 'Size_class', 'Beach', 'sample_id']]

    color_columns = [col for col in color_df.columns if col not in
                     ['Date_YYYY-MM-DD', 'Country_Region', 'Location_name',
                      'Location_lat', 'Location_lon', 'Transect', 'Position',
                      'Size_min_mm', 'Size_max_mm', 'Size_class', 'Beach', 'sample_id']]

    # Merge shape data
    merged_df = pd.merge(
        particle_df,
        shape_df[['sample_id'] + shape_columns],
        on='sample_id',
        how='left'
    )

    # Merge color data
    merged_df = pd.merge(
        merged_df,
        color_df[['sample_id'] + color_columns],
        on='sample_id',
        how='left'
    )

    # Ensure all shape and color columns are numeric
    for col in shape_columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)

    for col in color_columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)

    # Calculate proportions of different shapes
    shape_sum = merged_df[shape_columns].sum(axis=1)
    for col in shape_columns:
        merged_df[f'{col}_pct'] = np.where(shape_sum > 0,
                                           merged_df[col] / shape_sum * 100,
                                           0)

    # Calculate proportions of different colors
    color_sum = merged_df[color_columns].sum(axis=1)
    for col in color_columns:
        merged_df[f'{col}_pct'] = np.where(color_sum > 0,
                                           merged_df[col] / color_sum * 100,
                                           0)

    # Determine dominant shape and color
    def get_dominant_category(row, columns):
        max_val = 0
        max_cat = 'unknown'
        for col in columns:
            if pd.notna(row[col]) and row[col] > max_val:
                max_val = row[col]
                max_cat = col
        return max_cat

    merged_df['dominant_shape'] = merged_df.apply(
        lambda row: get_dominant_category(row, shape_columns), axis=1
    )

    merged_df['dominant_color'] = merged_df.apply(
        lambda row: get_dominant_category(row, color_columns), axis=1
    )

    # Create standardized shape category for DOER comparison
    shape_standardized = {
        'fragment': 'fragment',
        'foamed_plastic': 'foam',
        'pellet': 'pellet',
        'foil': 'foil',
        'fiber': 'fiber',
        'fibers': 'fiber',
        'fishing_line': 'fiber',
        'rope': 'fiber',
        'cigarette_butt': 'other',
        'rubber': 'other',
        'fabric': 'other',
        'unclear': 'unknown'
    }

    merged_df['Standardized_Shapes'] = merged_df['dominant_shape'].map(shape_standardized)

    # Map size classes to DOER categories
    size_mapping = {
        'microplastics': '<5 mm',
        'mesoplastics': '5-25 mm'
    }
    merged_df['Dominant_Size'] = merged_df['Size_class'].map(size_mapping)

    # Rename columns for consistency with DOER data
    merged_df = merged_df.rename(columns={
        'Beach': 'Waterbody',
        'Beach_Zone': 'Tidal_Zone',
        'particles_per_kg': 'mp/kg dw',
        'Location_name': 'Test_Area'
    })

    # Select relevant columns for integration with DOER data
    selected_columns = [
        'sample_id', 'Continent', 'Country', 'region', 'System', 'Waterbody',
        'Zone Area', 'Tidal_Zone', 'Test_Area', 'Year', 'Date',
        'mp/kg dw', 'log_concentration', 'Size_class',
        'Standardized_Shapes', 'Dominant_Size', 'dominant_color',
        'source', 'Season'
    ]

    return merged_df[selected_columns]


# Function to integrate datasets
def integrate_datasets(doer_df, taiwan_df):
    """
    Integrate DOER and Taiwan datasets.

    Args:
        doer_df: Processed DOER DataFrame
        taiwan_df: Processed Taiwan DataFrame

    Returns:
        Integrated DataFrame with standardized fields
    """
    print("Integrating datasets...")

    # Create a copy of each dataframe
    doer_copy = doer_df.copy()
    taiwan_copy = taiwan_df.copy()

    # Create a unique ID for DOER data
    doer_copy['sample_id'] = 'DOER_' + doer_copy['mpID'].astype(str).fillna('0')

    # Add missing columns to DOER data
    doer_copy['Date'] = pd.NaT
    doer_copy['Season'] = None
    doer_copy['Size_class'] = doer_copy['Dominant Size'].apply(
        lambda x: 'microplastics' if 'mm' in str(x) and any(d in str(x) for d in ['<1', '<2', '<3', '<5'])
        else 'mesoplastics' if 'mm' in str(x) and any(d in str(x) for d in ['1 to 5', '5'])
        else 'unknown'
    )

    # Rename DOER columns for consistency
    doer_copy = doer_copy.rename(columns={
        'Tidal Zone': 'Tidal_Zone',
        'Test Area': 'Test_Area',
        'Dominant Size': 'Dominant_Size'
    })

    # Add dominant color column to DOER (will be mostly unknown)
    doer_copy['dominant_color'] = 'unknown'

    # Select common columns for integration
    common_columns = [
        'sample_id', 'Continent', 'Country', 'region', 'System', 'Waterbody',
        'Zone Area', 'Tidal_Zone', 'Test_Area', 'Year', 'Date',
        'mp/kg dw', 'log_concentration', 'Size_class',
        'Standardized_Shapes', 'Dominant_Size', 'dominant_color',
        'source'
    ]

    # Concatenate datasets
    integrated_df = pd.concat(
        [doer_copy[common_columns], taiwan_copy[common_columns]],
        ignore_index=True
    )

    # Create environment type column
    def classify_environment(row):
        if row['System'] == 'Marine' and row['Zone Area'] == 'Beach':
            return 'Beach'
        elif row['System'] == 'Marine' and row['Zone Area'] == 'Coastal':
            return 'Coastal Marine'
        elif row['System'] == 'Estuarine':
            return 'Estuary'
        else:
            return row['System']

    integrated_df['environment_type'] = integrated_df.apply(classify_environment, axis=1)

    # Create standardized location column
    integrated_df['location'] = integrated_df.apply(
        lambda row: f"{row['Test_Area']}, {row['Country']}"
        if pd.notna(row['Test_Area']) else row['Country'],
        axis=1
    )

    # Classify concentration levels
    def classify_concentration(conc):
        if conc < 10:
            return 'Very Low'
        elif conc < 100:
            return 'Low'
        elif conc < 1000:
            return 'Medium'
        elif conc < 10000:
            return 'High'
        else:
            return 'Very High'

    integrated_df['concentration_level'] = integrated_df['mp/kg dw'].apply(classify_concentration)

    print(f"Integration complete. Final dataset has {len(integrated_df)} rows.")

    return integrated_df


# Function to generate summary visualizations
def generate_summary_visualizations(integrated_df, output_dir):
    """
    Generate summary visualizations for the integrated dataset.

    Args:
        integrated_df: Integrated DataFrame
        output_dir: Directory to save visualizations
    """
    print("Generating summary visualizations...")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Concentration comparison by environment
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=integrated_df,
        x='environment_type',
        y='log_concentration',
        hue='source',
        palette='Set2'
    )
    plt.title('Microplastic Concentrations by Environment Type', fontsize=14)
    plt.xlabel('Environment Type', fontsize=12)
    plt.ylabel('Log10(Particles per kg)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Data Source')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/concentration_by_environment.png', dpi=300)
    plt.close()

    # 2. Time series of concentrations (Taiwan data only)
    taiwan_df = integrated_df[integrated_df['source'] == 'Taiwan Beaches'].copy()
    taiwan_df = taiwan_df.sort_values('Date')

    if not taiwan_df.empty and pd.notna(taiwan_df['Date']).any():
        # Group by date and beach
        time_series = taiwan_df.groupby(['Date', 'Waterbody'])['mp/kg dw'].mean().reset_index()

        plt.figure(figsize=(14, 7))

        for beach in time_series['Waterbody'].unique():
            beach_data = time_series[time_series['Waterbody'] == beach]
            plt.plot(beach_data['Date'], beach_data['mp/kg dw'], marker='o', linewidth=2, label=beach)

        plt.title('Temporal Trends in Microplastic Concentration', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Particles per kg (dry weight)', fontsize=12)
        plt.legend(title='Beach')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temporal_trends.png', dpi=300)
        plt.close()

    # 3. Shape distribution
    shape_counts = integrated_df.groupby(['source', 'Standardized_Shapes']).size().reset_index(name='count')

    plt.figure(figsize=(12, 7))
    chart = sns.barplot(
        data=shape_counts,
        x='Standardized_Shapes',
        y='count',
        hue='source',
        palette='Set2'
    )
    plt.title('Distribution of Plastic Shapes by Data Source', fontsize=14)
    plt.xlabel('Shape Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Data Source')
    chart.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shape_distribution.png', dpi=300)
    plt.close()

    # 4. Beach zone distribution (Taiwan only)
    beach_zone_data = taiwan_df.groupby(['Tidal_Zone', 'Size_class'])['mp/kg dw'].mean().reset_index()

    plt.figure(figsize=(14, 8))
    chart = sns.barplot(
        data=beach_zone_data,
        x='Tidal_Zone',
        y='mp/kg dw',
        hue='Size_class',
        palette='Blues'
    )
    plt.title('Microplastic Concentration by Beach Zone', fontsize=14)
    plt.xlabel('Beach Zone', fontsize=12)
    plt.ylabel('Average Particles per kg', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Size Class')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/beach_zone_distribution.png', dpi=300)
    plt.close()

    print("Visualizations saved to", output_dir)


# Main function to run the entire process
def main():
    """Main function to process all datasets and generate visualizations."""
    import os

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