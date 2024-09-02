import pandas as pd
import numpy as np
import argparse
import os

def filter_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # List of columns to check for NaNs
    columns_to_check = ['id', 'latitude', 'longitude', 'land_cover', 'road_index', 
                        'drive_side', 'climate', 'soil', 'dist_sea', 'quadtree_10_1000']
    
    # Create a mask for rows where at least one of the specified columns is NaN
    mask = df[columns_to_check].isna().any(axis=1)
    
    # Count the number of rows that will be removed
    rows_to_remove = mask.sum()
    
    # Remove the rows where the mask is True
    df_filtered = df[~mask]
    
    # Save the filtered dataframe to a new CSV file
    df_filtered.to_csv(output_file, index=False)
    
    print(f"Processed {input_file}")
    print(f"Removed {rows_to_remove} rows with NaNs")
    print(f"Saved filtered data to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Filter CSV files to remove rows with NaNs in specific columns.")
    parser.add_argument("input", help="Input CSV file or directory")
    parser.add_argument("output", help="Output CSV file or directory")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        # Process a single file
        filter_csv(args.input, args.output)
    elif os.path.isdir(args.input):
        # Process all CSV files in the directory
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        for filename in os.listdir(args.input):
            if filename.endswith(".csv"):
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(args.output, f"filtered_{filename}")
                filter_csv(input_path, output_path)
    else:
        print("Input must be a file or directory")

if __name__ == "__main__":
    main()