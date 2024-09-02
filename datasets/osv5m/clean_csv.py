import os
import pandas as pd
import argparse
from tqdm import tqdm

def clean_csv(input_csv, output_csv, img_dir, id_column='id', img_extension='.jpg'):
    # Read the input CSV
    df = pd.read_csv(input_csv)
    
    # Function to check if image exists
    def image_exists(img_id):
        img_path = os.path.join(img_dir, f"{img_id}{img_extension}")
        return os.path.exists(img_path)
    
    # Apply the check to each row with a progress bar
    tqdm.pandas(desc="Checking images")
    df['image_exists'] = df[id_column].progress_apply(image_exists)
    
    # Filter the dataframe
    df_cleaned = df[df['image_exists']]
    
    # Remove the 'image_exists' column
    df_cleaned = df_cleaned.drop('image_exists', axis=1)
    
    # Save the cleaned dataframe
    df_cleaned.to_csv(output_csv, index=False)
    
    # Print summary
    print(f"Original number of rows: {len(df)}")
    print(f"Number of rows after cleaning: {len(df_cleaned)}")
    print(f"Number of rows removed: {len(df) - len(df_cleaned)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean CSV by removing rows with missing images.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("output_csv", help="Path to save the output CSV file")
    parser.add_argument("img_dir", help="Directory containing the images")
    parser.add_argument("--id_column", default="id", help="Name of the column containing image IDs")
    parser.add_argument("--img_extension", default=".jpg", help="Image file extension")
    
    args = parser.parse_args()
    
    clean_csv(args.input_csv, args.output_csv, args.img_dir, args.id_column, args.img_extension)

# Usage:
# python csv_image_cleanup.py input.csv output.csv /path/to/images --id_column image_id --img_extension .png