import lmdb
import pickle
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
import queue

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

def worker(queue, result_queue, image_dir, transform):
    while True:
        item = queue.get()
        if item is None:
            break
        idx, row = item
        image_path = os.path.join(image_dir, f"{row['id']}.jpg")
        img = process_image(image_path, transform)
        data = {
            'image': img,
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'quadtree_10_1000': row['quadtree_10_1000']
        }
        result_queue.put((idx, data))

def writer(queue, txn, num_samples):
    with tqdm(total=num_samples, desc="Writing Images") as pbar:
        while True:
            item = queue.get()
            if item is None:
                break
            idx, data = item
            txn.put(f'{idx}'.encode(), pickle.dumps(data))
            pbar.update(1)
        txn.put('length'.encode(), str(pbar.n).encode())
        

def create_lmdb_dataset(lmdb_path, image_dir, metadata_df, transform, num_workers=8):
    num_samples = len(metadata_df)
    env = lmdb.open(lmdb_path, map_size=1099511627776, writemap=True)
    
    work_queue = queue.Queue()
    result_queue = queue.Queue()

    with env.begin(write=True) as txn:
        # Start worker threads
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            workers = [executor.submit(worker, work_queue, result_queue, image_dir, transform) for _ in range(num_workers)]
            
            # Start writer thread
            writer_thread = threading.Thread(target=writer, args=(result_queue, txn, num_samples))
            writer_thread.start()

            # Feed work to workers
            for idx, row in metadata_df.iterrows():
                work_queue.put((idx, row))

            # Signal workers to finish
            for _ in range(num_workers):
                work_queue.put(None)

            # Wait for workers to finish and process results
            for future in workers:
                future.result()

            # Signal writer to finish
            result_queue.put(None)

            # Wait for writer to finish
            writer_thread.join()

    env.close()

# Usage
dataset_base = 'datasets/osv5m'
image_dir = f'{dataset_base}/images/train'
csv_path = f'{dataset_base}/train_extras_removed.csv'
lmdb_path = 'dataset_lmdb'

metadata_df = pd.read_csv(csv_path)
transform = train_transform()
create_lmdb_dataset(lmdb_path, image_dir, metadata_df, transform)
