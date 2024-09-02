import os
import pandas as pd
import numpy as np
import h5py
from PIL import Image

def create_hdf5(image_dir, annotation_path, output_path):
    df = pd.read_csv(annotation_path, index_col='id')
    with h5py.File(output_path, 'w') as hdf:
        for idx, image_name in enumerate(os.listdir(image_dir)):
            image_path = os.path.join(image_dir, image_name)
            info_id = os.path.splitext(image_name)[0]
            try:
                # Check if image file exists and is accessible
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                # Load image
                image = Image.open(image_path)
                image_np = np.array(image)

                # Store image in HDF5
                hdf.create_dataset(f"images/{info_id}", data=image_np, compression="gzip")

                # Check if info_id is in DataFrame
                if info_id not in df.index:
                    raise KeyError(f"ID {info_id} not found in DataFrame")

                # Store metadata in HDF5
                metadata_group = hdf.create_group(f"metadata/{info_id}")
                for column in df.columns:
                    metadata_group.create_dataset(column, data=np.array(df.at[info_id, column]))
            except Exception as e:
                print(f'Exception {e} for id: {info_id}, idx: {idx}, image_path: {image_path}')
                continue
# Usage
create_hdf5('datasets/OpenWorld/images/test', 'datasets/OpenWorld/test.csv', 'datasets/OpenWorld/test.hdf5')
