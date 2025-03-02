# Temporal Trends in Microplastics Visualization
# This script creates a publication-quality visualization showing how microplastic
# concentrations change over time at the Taiwan beaches

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import calendar

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def create_temporal_trends_visualization(particle_files, output_path):
    """
    Create a visualization showing temporal trends in microplastic concentrations.
    
    Args:
        particle_files: List of Taiwan particle count CSV files
        output_path: Path to save the visualization
    """
    # Load data
    taiwan_dfs = []
    
    for file_path in particle_files:
        beach_name = file_path.split('/')[1].split(' ')[0]
        df = pd.read_csv(file_path)
        df['Beach'] = beach_name
        taiwan_dfs.append(df)
    
    # Combine data
    df = pd.concat(taiwan_dfs, ignore_index=True)
    
    # Calculate concentration in particles per kg
    df['particles_per_kg'] = (
        df['Particle_count'] / (df['Weight_dry_sand_g'] / 1000)
    )
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date_YYYY-MM-DD'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month  # Month as int
    df['MonthName'] = df['Date'].dt.strftime('%b')
    
    # Drop any rows with missing dates
    df = df.dropna(subset=['Date'])
    
    # Print a sample of the data to debug
    print("First few rows of DataFrame:")
    print(df[['Beach', 'Date_YYYY-MM-DD', 'Date', 'Month']].head())
    
    # Create season column
    df['Season'] = pd.cut(
        df['Month'], 
        bins=[0, 3, 6, 9, 12], 
        labels=['Winter', 'Spring', 'Summer', 'Fall'],
        include_lowest=True
    )
    
    # Calculate monthly averages for each beach and size class
    monthly_avg = df.groupby(['Beach', 'Date', 'Season', 'Size_class'])['particles_per_kg'].mean().reset_index()
    
    # Calculate seasonal averages
    seasonal_avg = df.groupby(['Beach', 'Season', 'Size_class'])['particles_per_kg'].mean().reset_index()
    
    # Calculate beach zone averages by season
    zone_seasonal = df.groupby(['Beach', 'Season', 'Beach_Zone', 'Size_class'])['particles_per_kg'].mean().reset_index()
    
    # Create the visualization
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # 1. Time series plot
    ax1 = plt.subplot(gs[0, :])
    
    # Define colors for beaches and size classes
    beach_colors = {
        'Longmen': '#4C72B0',
        'Xialiao': '#DD8452'
    }
    
    size_markers = {
        'microplastics': 'o',
        'mesoplastics': '^'
    }
    
    # Filter for more readable plotting
    monthly_micro = monthly_avg[monthly_avg['Size_class'] == 'microplastics']
    
    # Plot time series for each beach
    for beach in monthly_micro['Beach'].unique():
        beach_data = monthly_micro[monthly_micro['Beach'] == beach]
        beach_name = beach.split('_')[0]  # Remove '_Beach' suffix
        
        # Plot the line
        ax1.plot(beach_data['Date'], beach_data['particles_per_kg'], 
                 marker='o', markersize=8, linestyle='-', linewidth=2, 
                 color=beach_colors[beach_name], label=f"{beach_name} Beach")
        
        # Add data points
        for _, row in beach_data.iterrows():
            ax1.scatter(row['Date'], row['particles_per_kg'], 
                       s=80, color=beach_colors[beach_name], 
                       edgecolor='white', linewidth=1, zorder=5)
    
    # Customize the time series plot
    ax1.set_title('Temporal Trends in Microplastic Concentration (2018-2019)', fontsize=16)
    ax1.set_ylabel('Microplastic Concentration (particles/kg)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Format x-axis to show months and years clearly
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add seasonal background shading
    season_spans = []
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = monthly_avg[monthly_avg['Season'] == season]
        if not season_data.empty:
            min_date = season_data['Date'].min()
            max_date = season_data['Date'].max()
            season_spans.append((season, min_date, max_date))
    
    # Define season colors (light, for background)
    season_colors = {
        'Winter': '#E1F5FE',  # Light blue
        'Spring': '#E8F5E9',  # Light green
        'Summer': '#FFFDE7',  # Light yellow
        'Fall': '#FBE9E7'     # Light orange
    }
    
    # Add background shading for seasons
    for season, min_date, max_date in season_spans:
        ax1.axvspan(min_date, max_date, alpha=0.3, color=season_colors[season])
    
    # Add legend
    ax1.legend(loc='upper left', frameon=True)
    
    # 2. Seasonal bar chart
    ax2 = plt.subplot(gs[1, 0])
    
    # Filter for microplastics
    seasonal_micro = seasonal_avg[seasonal_avg['Size_class'] == 'microplastics']
    
    # Order seasons correctly
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    
    # Create seasonal bar chart
    for i, beach in enumerate(seasonal_micro['Beach'].unique()):
        beach_data = seasonal_micro[seasonal_micro['Beach'] == beach]
        beach_name = beach.split('_')[0]
        
        # Reorder data by season
        ordered_data = pd.DataFrame(index=season_order)
        for _, row in beach_data.iterrows():
            ordered_data.loc[row['Season'], 'particles_per_kg'] = row['particles_per_kg']
        
        # Fill missing seasons with 0
        ordered_data = ordered_data.fillna(0)
        
        # Plot the bars
        x = np.arange(len(season_order))
        width = 0.35
        offset = width * (i - 0.5)
        
        ax2.bar(x + offset, ordered_data['particles_per_kg'], width, 
               label=f"{beach_name} Beach", color=beach_colors[beach_name], 
               edgecolor='white', linewidth=1)
    
    # Customize the seasonal bar chart
    ax2.set_title('Seasonal Patterns in Microplastic Concentration', fontsize=14)
    ax2.set_ylabel('Particles/kg', fontsize=12)
    ax2.set_xticks(np.arange(len(season_order)))
    ax2.set_xticklabels(season_order)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right')
    
    # 3. Heatmap of concentrations by zone and season
    ax3 = plt.subplot(gs[1, 1])
    
    # Filter for microplastics and reshape data for heatmap
    zone_micro = zone_seasonal[zone_seasonal['Size_class'] == 'microplastics']
    
    # Create separate heatmaps for each beach
    beaches = zone_micro['Beach'].unique()
    if len(beaches) > 1:
        # Create a separate heatmap for the first beach
        beach1_data = zone_micro[zone_micro['Beach'] == beaches[0]]
        
        # Pivot data for heatmap
        heatmap_data = beach1_data.pivot_table(
            index='Beach_Zone', 
            columns='Season', 
            values='particles_per_kg',
            aggfunc='mean'
        )
        
        # Reorder seasons and zones
        beach_zones = ['dune', 'backshore', 'storm_line', 'supra_littoral', 'high_tide_line', 'intertidal']
        heatmap_data = heatmap_data.reindex(index=[z for z in beach_zones if z in heatmap_data.index])
        heatmap_data = heatmap_data.reindex(columns=season_order)
        
        # Create the heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                   linewidths=0.5, ax=ax3, cbar_kws={'label': 'Particles/kg'})
        
        # Customize the heatmap
        ax3.set_title(f'Seasonal Zone Patterns ({beaches[0].split("_")[0]} Beach)', fontsize=14)
        ax3.set_ylabel('Beach Zone', fontsize=12)
        ax3.set_xlabel('Season', fontsize=12)
    
    # 4. Month-by-month comparison
    ax4 = plt.subplot(gs[2, :])
    
    # Group data by month for each beach
    monthly_summary = df.groupby(['Beach', 'Year', 'Month', 'MonthName', 'Size_class'])['particles_per_kg'].mean().reset_index()
    
    # Filter for microplastics
    monthly_micro_summary = monthly_summary[monthly_summary['Size_class'] == 'microplastics']
    
    # Create month names in order
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]
    
    # Create a dictionary to store data by beach
    beach_month_data = {}
    
    for beach in monthly_micro_summary['Beach'].unique():
        beach_data = monthly_micro_summary[monthly_micro_summary['Beach'] == beach]
        beach_name = beach.split('_')[0]
        
        # Initialize with zeros
        month_values = [0] * 12
        
        # Fill in available values
        for _, row in beach_data.iterrows():
            try:
                month_idx = int(row['Month']) - 1  # 0-based index, ensure it's an integer
                month_values[month_idx] = float(row['particles_per_kg'])
            except (ValueError, TypeError):
                print(f"Skipping invalid month value: {row['Month']}")
        
        beach_month_data[beach_name] = month_values
    
    # Set up x positions
    x = np.arange(len(month_names))
    width = 0.35
    
    # Plot bars for each beach
    beaches_list = list(beach_month_data.keys())
    for i, beach in enumerate(beaches_list):
        offset = width * (i - 0.5 + 0.5/len(beaches_list))
        ax4.bar(x + offset, beach_month_data[beach], width, 
               label=f"{beach} Beach", color=beach_colors[beach], 
               edgecolor='white', linewidth=1)
    
    # Customize the monthly chart
    ax4.set_title('Monthly Microplastic Concentrations', fontsize=14)
    ax4.set_xlabel('Month', fontsize=12)
    ax4.set_ylabel('Particles/kg', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(month_names)
    ax4.grid(axis='y', linestyle='--', alpha=0.3)
    ax4.legend(loc='upper right')
    
    # Add annotations for seasonal peaks and patterns
    # Find months with highest concentrations
    for beach in beaches_list:
        month_values = beach_month_data[beach]
        if any(month_values):  # Check if there are non-zero values
            max_month_idx = np.argmax(month_values)
            max_value = month_values[max_month_idx]
            
            if max_value > 0:
                ax4.annotate(
                    f"Peak: {max_value:.1f}",
                    xy=(max_month_idx, max_value),
                    xytext=(0, 20),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color='black'),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontsize=9
                )
    
    # Add explanatory text about seasonal patterns
    explanation_text = """
    Key Observations:
    • Highest concentrations typically observed in summer and fall
    • Storm line zones accumulate more microplastics during summer months
    • Seasonal patterns differ between beaches, suggesting varied inputs
    • Weather events may influence temporal distribution
    """
    
    # Add text box to the first plot
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax1.text(0.02, 0.98, explanation_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', bbox=props)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Temporal trends visualization saved to {output_path}")

if __name__ == "__main__":
    # Define file paths
    particle_files = [
        'data/Longmen Beach particle count.csv',
        'data/Xialiao Beach particle count.csv'
    ]
    
    output_path = 'outputs/temporal_trends.png'
    
    # Create visualization
    create_temporal_trends_visualization(particle_files, output_path)
