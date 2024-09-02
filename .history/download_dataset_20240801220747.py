from huggingface_hub import hf_hub_download

# for i in range(5):
#     hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/test", repo_type='dataset', local_dir="datasets/OpenWorld")
# hf_hub_download(repo_id="osv5m/osv5m", filename="README.md", repo_type='dataset', local_dir="datasets/OpenWorld")
    
# for i in range(10):
#     hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/train", repo_type='dataset', local_dir="datasets/OpenWorld")


import os
import zipfile
import shutil

# Define the root directory
root_dir = "datasets/OpenWorld"

# Walk through the directory
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".zip"):
            # Full path to the zip file
            zip_path = os.path.join(root, file)
            
            # Extract the zip file into a temporary directory
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                temp_extract_dir = os.path.join(root, "temp_extract")
                zip_ref.extractall(temp_extract_dir)
            
            # Move the contents of the extracted folder up one level
            for extracted_root, extracted_dirs, extracted_files in os.walk(temp_extract_dir):
                for extracted_file in extracted_files:
                    # Move file to the root directory
                    shutil.move(os.path.join(extracted_root, extracted_file), root)
                break  # Only process the top level of the extracted folder
            
            # Remove the temporary extraction directory
            shutil.rmtree(temp_extract_dir)
            
            # Remove the original zip file
            os.remove(zip_path)



# hf_hub_download(repo_id="osv5m/osv5m", filename="train.csv", repo_type='dataset', local_dir="datasets/OpenWorld")
# hf_hub_download(repo_id="osv5m/osv5m", filename="test.csv", repo_type='dataset', local_dir="datasets/OpenWorld")

hf_hub_download(repo_id="osv5m/osv5m", filename="osv5m.py", repo_type='dataset', local_dir="datasets/OpenWorld")