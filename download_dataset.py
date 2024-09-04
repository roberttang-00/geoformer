# from huggingface_hub import hf_hub_download

# for i in range(5):
#     hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/test", repo_type='dataset', local_dir="datasets/OpenWorld")
# hf_hub_download(repo_id="osv5m/osv5m", filename="README.md", repo_type='dataset', local_dir="datasets/OpenWorld")
    
# for i in range(10):
#     hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/train", repo_type='dataset', local_dir="datasets/OpenWorld")

from huggingface_hub import snapshot_download
snapshot_download(repo_id="osv5m/osv5m", local_dir="datasets/osv5m", repo_type='dataset')

import os
import zipfile
import shutil

root_dir = "datasets/osv5m"

for root, dirs, files in os.walk(root_dir):
    zip_files = [file for file in files if file.endswith(".zip")]

    if zip_files:
        for zip_file in zip_files:
            zip_path = os.path.join(root, zip_file)
            print(f"Processing {zip_path}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                print(f"Extracting to {root}")
                zip_ref.extractall(root)

            os.remove(zip_path)


# hf_hub_download(repo_id="osv5m/osv5m", filename="train.csv", repo_type='dataset', local_dir="datasets/OpenWorld")
# hf_hub_download(repo_id="osv5m/osv5m", filename="test.csv", repo_type='dataset', local_dir="datasets/OpenWorld")

# hf_hub_download(repo_id="osv5m/osv5m", filename="osv5m.py", repo_type='dataset', local_dir="datasets/OpenWorld")
