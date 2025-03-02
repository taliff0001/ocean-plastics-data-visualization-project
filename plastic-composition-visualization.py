# Plastic Types and Shapes Visualization
# This script creates a publication-quality visualization showing the composition
# of different plastic types found in the Taiwan beaches

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.transforms as mtransforms

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def create_plastic_composition_visualization(shape_files, color_files, output_path):
    """
    Create a visualization showing plastic composition and shapes.
    
    Args:
        shape_files: List of Taiwan shape count CSV files
        color_files: List of Taiwan color count CSV files
        output_path: Path to save the visualization
    """
    # Load shape data
    shape_dfs = []
    for file_path in shape_files:
        beach_name = file_path.split()[0]
        df = pd.read_csv(file_path)
        df['Beach'] = beach_name
        shape_dfs.append(df)
    
    shape_df = pd.concat(shape_dfs, ignore_index=True)
    
    # Load color data
    color_dfs = []
    for file_path in color_files:
        beach_name = file_path.split()[0]
        df = pd.read_csv(file_path)
        df['Beach'] = beach_name
        color_dfs.append(df)
    
    color_df = pd.concat(color_dfs, ignore_index=True)
    
    # Get shape and color columns
    shape_columns = [col for col in shape_df.columns if col not in 
                     ['Date_YYYY-MM-DD', 'Country_Region', 'Location_name', 
                      'Location_lat', 'Location_lon', 'Transect', 'Position', 
                      'Size_min_mm', 'Size_max_mm', 'Size_class', 'Beach']]
    
    color_columns = [col for col in color_df.columns if col not in 
                     ['Date_YYYY-MM-DD', 'Country_Region', 'Location_name', 
                      'Location_lat', 'Location_lon', 'Transect', 'Position', 
                      'Size_min_mm', 'Size_max_mm', 'Size_class', 'Beach']]
    
    # Calculate totals by shape and convert to long format for plotting
    shape_totals = shape_df.groupby(['Beach', 'Size_class'])[shape_columns].sum().reset_index()
    
    # Create percentages
    for beach in shape_totals['Beach'].unique():
        for size in shape_totals['Size_class'].unique():
            mask = (shape_totals['Beach'] == beach) & (shape_totals['Size_class'] == size)
            row_sum = shape_totals.loc[mask, shape_columns].sum(axis=1).values[0]
            if row_sum > 0:
                for col in shape_columns:
                    shape_totals.loc[mask, f'{col}_pct'] = shape_totals.loc[mask, col] / row_sum * 100
    
    # Calculate totals by color
    color_totals = color_df.groupby(['Beach', 'Size_class'])[color_columns].sum().reset_index()
    
    # Create percentages
    for beach in color_totals['Beach'].unique():
        for size in color_totals['Size_class'].unique():
            mask = (color_totals['Beach'] == beach) & (color_totals['Size_class'] == size)
            row_sum = color_totals.loc[mask, color_columns].sum(axis=1).values[0]
            if row_sum > 0:
                for col in color_columns:
                    color_totals.loc[mask, f'{col}_pct'] = color_totals.loc[mask, col] / row_sum * 100
    
    # Create the visualization
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
    
    # 1. Shape composition by beach
    ax1 = plt.subplot(gs[0, :2])
    
    # Prepare data for stacked bar chart
    beaches = shape_totals['Beach'].unique()
    size_classes = shape_totals['Size_class'].unique()
    
    # Define colors for shapes
    shape_colors = {
        'fragment': '#4C72B0',       # Blue
        'foamed_plastic': '#DD8452',  # Orange
        'pellet': '#55A868',          # Green
        'foil': '#C44E52',            # Red
        'fiber': '#8172B3',           # Purple
        'fibers': '#8172B3',          # Purple
        'fishing_line': '#937860',    # Brown
        'cigarette_butt': '#DA8BC3',  # Pink
        'rope': '#8C8C8C',            # Gray
        'rubber': '#CCB974',          # Yellow
        'fabric': '#64B5CD',          # Light blue 
        'unclear': '#AAAAAA'          # Light gray
    }
    
    # Plot stacked bars for microplastics
    x = np.arange(len(beaches))
    width = 0.35
    
    # Filter for microplastics
    micro_shapes = shape_totals[shape_totals['Size_class'] == 'microplastics']
    
    bottom = np.zeros(len(beaches))
    for shape in shape_columns:
        shape_data = []
        for beach in beaches:
            beach_data = micro_shapes[micro_shapes['Beach'] == beach]
            value = beach_data[shape].values[0] if not beach_data.empty else 0
            shape_data.append(value)
        
        # Only plot if there are non-zero values
        if sum(shape_data) > 0:
            ax1.bar(x, shape_data, width, label=shape, bottom=bottom, 
                   color=shape_colors.get(shape, '#AAAAAA'))
            bottom += np.array(shape_data)
    
    # Create simplified beach names for display
    beach_names = [b.split('_')[0] + '\nMicroplastics' for b in beaches]
    
    # Customize the shape composition plot
    ax1.set_title('Plastic Type Composition by Beach', fontsize=16)
    ax1.set_ylabel('Number of Particles', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(beach_names)
    ax1.legend(title='Plastic Type', loc='upper right')
    
    # 2. Pie charts for shape composition percentages
    ax2 = plt.subplot(gs[0, 2])
    
    # Filter for microplastics from Xialiao beach (which has better shape data)
    xialiao_micro = shape_totals[(shape_totals['Beach'] == 'Xialiao_Beach') & 
                                 (shape_totals['Size_class'] == 'microplastics')]
    
    # Get percentages for pie chart
    shape_pcts = {}
    for shape in shape_columns:
        if not xialiao_micro.empty and shape in xialiao_micro.columns:
            value = xialiao_micro[shape].values[0]
            shape_pcts[shape] = value
    
    # Filter small values
    total = sum(shape_pcts.values())
    shape_pcts_filtered = {k: v for k, v in shape_pcts.items() if v/total > 0.01}  # Filter < 1%
    
    # Add "Other" category if needed
    other_sum = total - sum(shape_pcts_filtered.values())
    if other_sum > 0:
        shape_pcts_filtered['other'] = other_sum
    
    # Create pie chart
    wedges, texts, autotexts = ax2.pie(
        shape_pcts_filtered.values(), 
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=[shape_colors.get(k, '#AAAAAA') for k in shape_pcts_filtered.keys()]
    )
    
    # Make percentage labels more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
    
    # Create nicer labels
    shape_nice_names = {
        'fragment': 'Fragments',
        'foamed_plastic': 'Foam',
        'pellet': 'Pellets',
        'foil': 'Films/Foils',
        'fiber': 'Fibers',
        'fibers': 'Fibers',
        'fishing_line': 'Fishing Line',
        'cigarette_butt': 'Cigarette Butts',
        'rope': 'Rope',
        'rubber': 'Rubber',
        'fabric': 'Fabric',
        'unclear': 'Unidentified',
        'other': 'Other'
    }
    
    # Create legend with percentages
    legend_labels = [f"{shape_nice_names.get(shape, shape)} ({value/total*100:.1f}%)" 
                     for shape, value in shape_pcts_filtered.items()]
    
    ax2.legend(wedges, legend_labels, title="Plastic Types", 
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    ax2.set_title('Microplastic Type Distribution\n(Xialiao Beach)', fontsize=14)
    
    # 3. Color composition
    ax3 = plt.subplot(gs[1, 0])
    
    # Define colors for each category
    color_display_colors = {
        'no_color': '#FFFFFF',         # White
        'black': '#000000',            # Black
        'grey': '#808080',             # Gray
        'red_pink': '#FF5252',         # Red-Pink
        'orange_brown_yellow': '#FFA726', # Orange-Brown-Yellow
        'green': '#66BB6A',            # Green
        'blue': '#42A5F5',             # Blue
        'purple': '#AB47BC'            # Purple
    }
    
    # Filter for microplastics
    micro_colors = color_totals[color_totals['Size_class'] == 'microplastics']
    
    # Prepare data for horizontal bar chart
    color_data = []
    
    for beach in beaches:
        beach_colors = micro_colors[micro_colors['Beach'] == beach]
        if not beach_colors.empty:
            beach_name = beach.split('_')[0]
            
            for color in color_columns:
                if color in beach_colors.columns:
                    value = beach_colors[color].values[0]
                    pct = beach_colors[f'{color}_pct'].values[0] if f'{color}_pct' in beach_colors.columns else 0
                    
                    if not np.isnan(value) and value > 0:
                        color_data.append({
                            'Beach': beach_name,
                            'Color': color,
                            'Count': value,
                            'Percentage': pct
                        })
    
    # Convert to DataFrame for easier plotting
    color_df = pd.DataFrame(color_data)
    
    # Create nicer color names
    color_nice_names = {
        'no_color': 'Clear/Transparent',
        'black': 'Black',
        'grey': 'Gray',
        'red_pink': 'Red/Pink',
        'orange_brown_yellow': 'Orange/Brown/Yellow',
        'green': 'Green',
        'blue': 'Blue',
        'purple': 'Purple'
    }
    
    color_df['Color_Nice'] = color_df['Color'].map(color_nice_names)
    
    # Plot horizontal stacked bars
    beaches_list = color_df['Beach'].unique()
    colors_list = color_df['Color'].unique()
    
    # Create a pivot table
    pivot_data = color_df.pivot_table(
        index='Beach', 
        columns='Color', 
        values='Percentage',
        aggfunc='sum'
    ).fillna(0)
    
    # Plot the stacked bars
    pivot_data.plot(
        kind='barh', 
        stacked=True,
        color=[color_display_colors.get(c, '#AAAAAA') for c in pivot_data.columns],
        ax=ax3,
        edgecolor='white',
        linewidth=0.5
    )
    
    # Customize the color composition plot
    ax3.set_title('Color Distribution of Microplastics', fontsize=14)
    ax3.set_xlabel('Percentage (%)', fontsize=12)
    ax3.set_ylabel('')
    
    # Create custom legend with nicer labels
    handles = [Patch(facecolor=color_display_colors.get(color, '#AAAAAA'), 
                     edgecolor='white', label=color_nice_names.get(color, color)) 
               for color in pivot_data.columns]
    
    ax3.legend(handles=handles, title='Color', loc='lower right')
    
    # 4. Marine debris sources
    ax4 = plt.subplot(gs[1, 1:])
    
    # Create a table visualization showing likely sources of different plastic types
    source_data = {
        'Plastic Type': ['Fragments', 'Foam', 'Pellets', 'Films/Foils', 'Fibers', 'Fishing Line'],
        'Likely Sources': [
            'Breakdown of larger plastics, packaging',
            'Food containers, packaging, construction',
            'Industrial pre-production pellets, microbeads',
            'Plastic bags, wrappers, packaging',
            'Clothing, textiles, nets',
            'Fishing activities, recreational fishing'
        ],
        'Environmental Impact': [
            'Long persistence, ingestion by marine life',
            'Rapid fragmentation, toxic leaching',
            'Easily ingested by small organisms',
            'Entanglement, ingestion, breakdown to microplastics',
            'Microfiber shedding, water column suspension',
            'Entanglement, ghost fishing'
        ],
        'Proportion': [
            shape_pcts.get('fragment', 0) / total * 100 if total > 0 else 0,
            shape_pcts.get('foamed_plastic', 0) / total * 100 if total > 0 else 0,
            shape_pcts.get('pellet', 0) / total * 100 if total > 0 else 0,
            shape_pcts.get('foil', 0) / total * 100 if total > 0 else 0,
            (shape_pcts.get('fiber', 0) + shape_pcts.get('fibers', 0)) / total * 100 if total > 0 else 0,
            shape_pcts.get('fishing_line', 0) / total * 100 if total > 0 else 0
        ]
    }
    
    source_df = pd.DataFrame(source_data)
    
    # Sort by proportion
    source_df = source_df.sort_values('Proportion', ascending=False)
    
    # Create the table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Format proportions as percentages
    source_df['Proportion'] = source_df['Proportion'].apply(lambda x: f"{x:.1f}%")
    
    table = ax4.table(
        cellText=source_df.values,
        colLabels=source_df.columns,
        loc='center',
        cellLoc='left',
        colColours=['#f2f2f2'] * len(source_df.columns),
        cellColours=[['#ffffff', '#ffffff', '#ffffff', '#f9f9f9']] * len(source_df)
    )
    
    # Customize table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the table header
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_height(0.2)
        
        # Add some padding
        cell.PAD = 0.05
    
    # Add a title
    ax4.set_title('Sources and Environmental Impact of Plastic Types', fontsize=14, pad=20)
    
    # Add environmental impact information
    info_text = """
    Environmental Significance:
    • Different plastic shapes indicate different sources and transport mechanisms
    • Foam dominates Taiwan beaches, suggesting land-based sources like food packaging
    • Pellets indicate industrial spills and shipping-related sources
    • Fragments represent the breakdown of larger items, showing long-term accumulation
    • Color can indicate weathering and time in the environment (clear/white often newer)
    """
    
    # Use transform to position text in the right coordinate system
    trans = mtransforms.blended_transform_factory(ax1.transAxes, ax1.transAxes)
    ax1.text(0.02, 0.02, info_text, transform=trans, fontsize=10, va='bottom', ha='left',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plastic composition visualization saved to {output_path}")

if __name__ == "__main__":
    # Define file paths
    shape_files = [
        'data/Longmen Beach shape count.csv',
        'data/Xialiao Beach shape count.csv'
    ]
    
    color_files = [
        'data/Longmen Beach color count.csv',
        'data/Xialiao Beach color count.csv'
    ]
    
    output_path = 'outputs/shape_distribution.png'
    
    # Create visualization
    create_plastic_composition_visualization(shape_files, color_files, output_path)
