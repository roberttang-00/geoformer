from huggingface_hub import hf_hub_download

# for i in range(5):
#     hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/test", repo_type='dataset', local_dir="datasets/OpenWorld")
# hf_hub_download(repo_id="osv5m/osv5m", filename="README.md", repo_type='dataset', local_dir="datasets/OpenWorld")
    
# for i in range(10):
#     hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/train", repo_type='dataset', local_dir="datasets/OpenWorld")


import os
import zipfile
import shutil

root_dir = "datasets/osv5m"

for root, dirs, files in os.walk(root_dir):
    temp_extract_dir = os.path.join(root, "temp_extract")
    has_one = False
    for file in files:
        if file.endswith(".zip"):
            has_one = True
            zip_path = os.path.join(root, file)
            print(f"Processing {zip_path}")
            
            # Extract the zip file into a temporary directory
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                print(f"Extracting to {temp_extract_dir}")
                zip_ref.extractall(temp_extract_dir)

            os.remove(zip_path)
    
    if has_one:
        for extracted_root, extracted_dirs, extracted_files in os.walk(temp_extract_dir):
            for extracted_file in extracted_files:
                print(f"Moving {os.path.join(extracted_root, extracted_file)} to {root}")
                shutil.move(os.path.join(extracted_root, extracted_file), root)
        
        shutil.rmtree(temp_extract_dir)



# hf_hub_download(repo_id="osv5m/osv5m", filename="train.csv", repo_type='dataset', local_dir="datasets/OpenWorld")
# hf_hub_download(repo_id="osv5m/osv5m", filename="test.csv", repo_type='dataset', local_dir="datasets/OpenWorld")

hf_hub_download(repo_id="osv5m/osv5m", filename="osv5m.py", repo_type='dataset', local_dir="datasets/OpenWorld")