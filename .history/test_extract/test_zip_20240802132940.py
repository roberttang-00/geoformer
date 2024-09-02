import os
import zipfile
import shutil

# Define the root directory
root_dir = "datasets/osv5m"

# Walk through the directory
for root, dirs, files in os.walk(root_dir):
    # root would be train/test
    for file in files:
        if file.endswith(".zip"):
            # Full path to the zip file
            zip_path = os.path.join(root, file)
            print(f"Processing {zip_path}")
            
            # Extract the zip file into a temporary directory
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                temp_extract_dir = os.path.join(root, "temp_extract")
                print(f"Extracting to {temp_extract_dir}")
                zip_ref.extractall(temp_extract_dir)

            os.remove(zip_path)
    
            # Move the contents of the extracted folder up one level
            for extracted_root, extracted_dirs, extracted_files in os.walk(root):
                for extracted_file in extracted_files:
                    print(f"Moving from {os.path.join(extracted_root, extracted_file)} to {root}")
                    shutil.move(os.path.join(extracted_root, extracted_file), root)
            break