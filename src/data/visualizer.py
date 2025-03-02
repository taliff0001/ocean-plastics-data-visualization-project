"""Data visualization utilities for ocean plastics analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


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