# Global Microplastics Comparison Visualization
# This script creates a publication-quality visualization comparing microplastic concentrations
# between the Taiwan beaches and global data from the DOER database

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def create_global_comparison_visualization(doer_file, taiwan_particle_files, output_path):
    """
    Create a visualization comparing Taiwan beach microplastic concentrations with global data.
    
    Args:
        doer_file: Path to DOER database CSV
        taiwan_particle_files: List of Taiwan particle count CSV files
        output_path: Path to save the visualization
    """
    # Load DOER data
    doer_df = pd.read_csv(doer_file)
    
    # Filter for coastal and marine environments
    env_filter = doer_df['System'].isin(['Marine', 'Estuarine'])
    coastal_filter = doer_df['Zone Area'].str.contains('Coastal|Beach', na=False)
    doer_filtered = doer_df[env_filter & coastal_filter].copy()
    
    # Create region column
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
    
    doer_filtered['region'] = doer_filtered.apply(
        lambda row: assign_region(row['Continent'], row['Country']), axis=1
    )
    
    # Load Taiwan data
    taiwan_dfs = []
    for file_path in taiwan_particle_files:
        beach_name = file_path.split('/')[1].split(' ')[0]
        df = pd.read_csv(file_path)
        df['Beach'] = beach_name
        taiwan_dfs.append(df)
    
    taiwan_df = pd.concat(taiwan_dfs, ignore_index=True)
    
    # Calculate concentration in particles per kg
    taiwan_df['mp/kg dw'] = (
        taiwan_df['Particle_count'] / (taiwan_df['Weight_dry_sand_g'] / 1000)
    )
    
    # Add region and source columns
    taiwan_df['region'] = 'Taiwan'
    taiwan_df['source'] = 'Taiwan Beaches'
    doer_filtered['source'] = 'DOER Database'
    
    # Create log-transformed concentrations
    doer_filtered['log_concentration'] = np.log10(doer_filtered['mp/kg dw'] + 1)
    taiwan_df['log_concentration'] = np.log10(taiwan_df['mp/kg dw'] + 1)
    
    # Calculate statistics by region
    doer_stats = doer_filtered.groupby('region')['mp/kg dw'].agg(['mean', 'median', 'std', 'count']).reset_index()
    
    # Calculate Taiwan statistics
    taiwan_stats = taiwan_df.groupby('Beach')['mp/kg dw'].agg(['mean', 'median', 'std', 'count']).reset_index()
    taiwan_stats['region'] = 'Taiwan'
    
    # Create the visualization
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1.5])
    
    # 1. Global comparison boxplot
    ax1 = plt.subplot(gs[0, :])
    
    # Combine data for the boxplot
    doer_plot = doer_filtered[['region', 'mp/kg dw', 'log_concentration', 'source']]
    
    # For Taiwan, filter to just microplastics for fair comparison
    taiwan_plot = taiwan_df[taiwan_df['Size_class'] == 'microplastics'][['region', 'mp/kg dw', 'log_concentration', 'source']]
    
    # Combine datasets for plotting
    combined_plot = pd.concat([doer_plot, taiwan_plot], ignore_index=True)
    
    # Create custom region order with Taiwan last (for emphasis)
    regions = combined_plot['region'].unique().tolist()
    if 'Taiwan' in regions:
        regions.remove('Taiwan')
    regions.append('Taiwan')  # Add Taiwan at the end
    
    # Create the boxplot with a logarithmic scale
    sns.boxplot(
        data=combined_plot, 
        x='region', 
        y='mp/kg dw',
        order=regions,
        hue='source',
        palette={'DOER Database': '#5790C0', 'Taiwan Beaches': '#E36262'},
        ax=ax1
    )
    
    # Set logarithmic scale for y-axis
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    
    # Customize the plot
    ax1.set_title('Microplastic Concentrations in Marine Environments', fontsize=16)
    ax1.set_xlabel('Region', fontsize=14)
    ax1.set_ylabel('Microplastic Concentration (particles/kg)', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add a horizontal line for global average
    global_mean = doer_filtered['mp/kg dw'].mean()
    ax1.axhline(y=global_mean, color='black', linestyle='--', alpha=0.7)
    ax1.text(0.02, global_mean*1.2, f'Global Mean: {global_mean:.1f}', 
             transform=ax1.get_yaxis_transform(), fontsize=10)
    
    # Adjust legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, title='Data Source', loc='upper right')
    
    # 2. Create a map visualization showing the Taiwan focus area
    ax2 = plt.subplot(gs[1, 0])
    
    # Create a simple world map visualization
    # For a full application, you would need to use a proper mapping library like geopandas
    # This is a simplified representation
    
    # Create a basic map using an image
    ax2.set_xlim(60, 180)
    ax2.set_ylim(0, 60)
    
    # Add country outlines (simplified)
    # East Asia region approximate outlines
    asia_x = [100, 130, 150, 140, 120, 100]
    asia_y = [5, 10, 30, 40, 50, 40]
    ax2.fill(asia_x, asia_y, color='#D3D3D3', alpha=0.5)
    
    # Mark Taiwan location
    taiwan_x, taiwan_y = 121.5, 24.0
    ax2.scatter(taiwan_x, taiwan_y, color='#E36262', s=100, zorder=5, 
                edgecolor='white', linewidth=1.5)
    
    # Add Taiwan label
    ax2.text(taiwan_x + 3, taiwan_y, 'Taiwan', fontsize=12, color='black',
            ha='left', va='center', fontweight='bold')
    
    # Mark other sampling locations with lower opacity
    # Sample DOER database locations in Asia
    locations = [
        ('South China Sea', 115, 15),
        ('Yellow Sea', 123, 38),
        ('Japan', 140, 35),
        ('South Korea', 127, 37),
        ('Southeast Asia', 105, 10)
    ]
    
    for name, x, y in locations:
        ax2.scatter(x, y, color='#5790C0', s=50, alpha=0.7, edgecolor='white')
        ax2.text(x + 1, y, name, fontsize=8, alpha=0.7, ha='left', va='center')
    
    # Customize the map
    ax2.set_title('Study Locations in East Asia', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xlabel('Longitude (°E)', fontsize=10)
    ax2.set_ylabel('Latitude (°N)', fontsize=10)
    
    # 3. Additional insights panel
    ax3 = plt.subplot(gs[1, 1])
    
    # Calculate Taiwan vs global statistics
    taiwan_mean = taiwan_plot['mp/kg dw'].mean()
    taiwan_median = taiwan_plot['mp/kg dw'].median()
    global_median = doer_filtered['mp/kg dw'].median()
    
    # Calculate how many times higher Taiwan is compared to global average
    ratio_mean = taiwan_mean / global_mean
    ratio_median = taiwan_median / global_median
    
    # Define a function to create a statistic bar
    def add_stat_bar(ax, y_pos, taiwan_val, global_val, label, color):
        bar_height = 0.3
        max_width = 10  # Maximum width for scaling
        
        # Calculate bar widths based on values
        taiwan_width = min(taiwan_val / max(taiwan_val, global_val) * max_width, max_width)
        global_width = min(global_val / max(taiwan_val, global_val) * max_width, max_width)
        
        # Draw bars
        ax.barh([y_pos + bar_height/2], [taiwan_width], height=bar_height, 
                color=color, alpha=0.7, label='Taiwan')
        ax.barh([y_pos - bar_height/2], [global_width], height=bar_height, 
                color='gray', alpha=0.5, label='Global')
        
        # Add values at the end of bars
        ax.text(taiwan_width + 0.2, y_pos + bar_height/2, f"{taiwan_val:.1f}", 
                va='center', fontsize=9, color=color)
        ax.text(global_width + 0.2, y_pos - bar_height/2, f"{global_val:.1f}", 
                va='center', fontsize=9, color='gray')
        
        # Add label on y-axis
        ax.text(-0.5, y_pos, label, ha='right', va='center', fontsize=10, fontweight='bold')
    
    # Add title
    ax3.text(0.5, 0.95, 'Taiwan vs Global Comparison', 
             ha='center', va='top', fontsize=14, fontweight='bold',
             transform=ax3.transAxes)
    
    # Add statistic bars
    add_stat_bar(ax3, 3, taiwan_mean, global_mean, 'Mean', '#E36262')
    add_stat_bar(ax3, 2, taiwan_median, global_median, 'Median', '#E36262')
    
    # Add comparison text
    ax3.text(0.5, 0.7, f"Taiwan beaches have {ratio_mean:.1f}x higher\nmicroplastic concentrations\nthan the global average", 
             ha='center', va='center', fontsize=12, 
             transform=ax3.transAxes,
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add information about sample sizes
    ax3.text(0.1, 0.4, 
             f"Sample Sizes:\nTaiwan: {len(taiwan_plot)} samples\nGlobal: {len(doer_plot)} samples", 
             ha='left', va='center', fontsize=10, 
             transform=ax3.transAxes)
    
    # Add note about methodology
    ax3.text(0.1, 0.2, 
             "Note: Only comparable marine and coastal\nsamples from DOER database used.\nTaiwan data is for microplastics (1-5mm).", 
             ha='left', va='center', fontsize=8, style='italic',
             transform=ax3.transAxes,
             bbox=dict(facecolor='#f5f5f5', alpha=0.5, boxstyle='round,pad=0.3'))
    
    # Customize the insights panel
    ax3.set_xlim(-1, 11)
    ax3.set_ylim(0, 4)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Global comparison visualization saved to {output_path}")

if __name__ == "__main__":
    # Define file paths
    doer_file = 'data/DOER_microplastics_database.csv'
    
    taiwan_particle_files = [
        'data/Longmen Beach particle count.csv',
        'data/Xialiao Beach particle count.csv'
    ]
    
    output_path = 'outputs/concentration_by_environment.png'
    
    # Create visualization
    create_global_comparison_visualization(doer_file, taiwan_particle_files, output_path)
