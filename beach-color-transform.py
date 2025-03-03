import pandas as pd

# Load the dataset
df = pd.read_csv("combined_beach_color_count.csv")

# Display original dataset info
print("Original dataset info:")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Columns: {', '.join(df.columns)}")
print()

# Step 1: Create a new column named "Dominant Color" and initialize it with "unknown"
df["Dominant Color"] = "unknown"
print("Step 1: Added 'Dominant Color' column with default value 'unknown'")

# Step 2: Define a function to find the dominant color for each row
def find_dominant_color(row):
    """
    Find the column with the highest numerical value among the color columns.
    If the highest value is 0, keep "unknown".
    If the column is "no_color", return "transparent".
    Otherwise, return the name of the column.
    """
    # List of color columns to check
    color_cols = ['no_color', 'black', 'grey', 'red_pink',
                 'orange_brown_yellow', 'green', 'blue', 'purple']
    
    # Extract the color values for this row
    color_values = [row[col] for col in color_cols]
    
    # Find the maximum value and its index
    max_val = max(color_values)
    max_idx = color_values.index(max_val)
    
    # If the maximum value is 0, keep "unknown"
    if max_val == 0:
        return "unknown"
    
    # Get the name of the column with the maximum value
    max_col = color_cols[max_idx]
    
    # Replace "no_color" with "transparent"
    if max_col == "no_color":
        return "transparent"
    else:
        return max_col

# Apply the function to each row
df["Dominant Color"] = df.apply(find_dominant_color, axis=1)
print("Step 2: Applied function to determine the dominant color for each row")

# Display information about the dominant color transformation
print("\nDominant color distribution:")
print(df["Dominant Color"].value_counts())
print()

# Step 3: Drop the numerical color columns
color_cols = ['no_color', 'black', 'grey', 'red_pink',
             'orange_brown_yellow', 'green', 'blue', 'purple']

df = df.drop(columns=color_cols)
print("Step 3: Dropped numerical color columns")

# Step 4: Display information about the transformations and save the modified dataset
print("\nFinal dataset info:")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Columns: {', '.join(df.columns)}")
print()

# Save the modified dataset to a CSV file
output_file = "beach_data_with_dominant_color.csv"
df.to_csv(output_file, index=False)
print(f"Step 4: Modified dataset saved to '{output_file}'")

# Print a sample of the transformed data
print("\nSample of transformed data (first 5 rows):")
print(df.head(5).to_string())
