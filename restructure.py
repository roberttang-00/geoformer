import os
import shutil

def restructure_dataset(base_path):
    # List all subdirectories
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_path, subdir)
        
        # Move all files from subdirectory to base directory
        for file in os.listdir(subdir_path):
            src = os.path.join(subdir_path, file)
            dst = os.path.join(base_path, file)
            shutil.move(src, dst)
        
        # Remove the now-empty subdirectory
        os.rmdir(subdir_path)
    
    print("Dataset restructured successfully!")

# Usage
base_path = "/media/robert/1E4286EC4286C7CB/OldPC/0000/Geoformer/datasets/OpenWorld/images/train"
restructure_dataset(base_path)