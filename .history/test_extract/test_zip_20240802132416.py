import os
import zipfile
import shutil

# Define the root directory
root_dir = "datasets/osv5m"

# Walk through the directory
for root, dirs, files in os.walk(root_dir):
    # root would be train/test
    print(root, dirs, files)
    for file in files:
        if file.endswith(".zip"):
            # Full path to the zip file
            zip_path = os.path.join(root, file)
            
            # Extract the zip file into a temporary directory
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(root)

            os.remove(zip_path)
            
    # Move the contents of the extracted folder up one level
    for extracted_root, extracted_dirs, extracted_files in os.walk(root):
        for extracted_file in extracted_files:
            shutil.move(os.path.join(extracted_root, extracted_file), root)
    break