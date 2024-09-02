import h5py
import os
import pandas as pd
from PIL import Image
import io
import threading
from PIL import ImageFilter, ImageOps
import random
import torchvision.transforms as T 
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = T.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img

def train_transform():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomChoice([gray_scale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)]),
        T.ToTensor(),
        T.Lambda(lambda x: (x * 255).byte())
    ])

def process_image(image_path, transform):
    with Image.open(image_path) as img:
        img = transform(img)
    return np.array(img)

def worker_task(image_dir, metadata_df, hdf5_file, start_idx, end_idx, transform, lock, pbar):
    chunk_size = 250 # Smaller chunks for more frequent updates
    
    for chunk_start in range(start_idx, end_idx, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_idx)
        chunk_size_actual = chunk_end - chunk_start
        
        images = []
        latitudes = []
        longitudes = []
        quadtrees = []

        for idx in range(chunk_start, chunk_end):
            row = metadata_df.iloc[idx]
            image_path = os.path.join(image_dir, f"{row['id']}.jpg")
            img = process_image(image_path, transform)
            
            images.append(img)
            latitudes.append(row['latitude'])
            longitudes.append(row['longitude'])
            quadtrees.append(row['quadtree_10_1000'])

        with lock:
            hdf5_file['images'][chunk_start:chunk_end] = images
            hdf5_file['latitude'][chunk_start:chunk_end] = latitudes
            hdf5_file['longitude'][chunk_start:chunk_end] = longitudes
            hdf5_file['quadtree_10_1000'][chunk_start:chunk_end] = quadtrees
            pbar.update(chunk_size_actual)

def create_hdf5_dataset(hdf5_path, image_dir, metadata_df, transform, num_workers=8):
    num_samples = len(metadata_df)
    img_shape = (3, 224, 224)
    
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        hdf5_file.create_dataset('images', shape=(num_samples,) + img_shape, dtype=np.uint8, chunks=(100,) + img_shape)
        hdf5_file.create_dataset('latitude', shape=(num_samples,), dtype=np.float32, chunks=(100,))
        hdf5_file.create_dataset('longitude', shape=(num_samples,), dtype=np.float32, chunks=(100,))
        hdf5_file.create_dataset('quadtree_10_1000', shape=(num_samples,), dtype=np.int16, chunks=(100,))

        lock = threading.Lock()
        pbar = tqdm(total=num_samples, desc="Processing Images")

        chunk_size = num_samples // num_workers
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < num_workers - 1 else num_samples
                future = executor.submit(worker_task, image_dir, metadata_df, hdf5_file, start_idx, end_idx, transform, lock, pbar)
                futures.append(future)

            # Wait for all tasks to complete
            for future in futures:
                future.result()

        pbar.close()

if __name__ == "__main__":
    dataset_base = 'datasets/OpenWorld'
    image_dir = f'{dataset_base}/images/train'
    csv_path = f'{dataset_base}/train_extras_removed.csv'
    hdf5_path = 'dataset.hdf5'

    metadata_df = pd.read_csv(csv_path)
    transform = train_transform()
    create_hdf5_dataset(hdf5_path, image_dir, metadata_df, transform)