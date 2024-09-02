import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm

class CustomImagePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data):
        super(CustomImagePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.readers.File(file_root=data['image_dir'], files=data['image_files'])
        self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=224, random_area=(0.7, 1.0))
        self.flip = ops.Flip(device="gpu")
        self.gaussian_blur = ops.GaussianBlur(device="gpu", window_size=7)
        self.solarize = ops.ColorTwist(device="gpu")
        self.grayscale = ops.ColorSpaceConversion(device="gpu", image_type=types.RGB, output_type=types.GRAY)
        self.uniform = ops.Uniform(range=(0, 1))
        self.coin_flip = ops.CoinFlip(probability=0.2)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.res(images)
        images = self.flip(images)

        # Randomly apply one of Gaussian Blur, Solarize, or Grayscale
        condition = fn.random.uniform(range=(0, 1)) 
        gaussian_applied = fn.cast(condition < 0.33, dtype=types.FLOAT)
        solarize_applied = fn.cast((condition >= 0.33) & (condition < 0.66), dtype=types.FLOAT)
        grayscale_applied = fn.cast(condition >= 0.66, dtype=types.FLOAT)

        blurred = self.gaussian_blur(images)
        solarized = self.solarize(images, brightness=fn.random.uniform(range=(0.5, 2)))
        gray = self.grayscale(images)
        gray = fn.cat(gray, gray, gray, axis=2)  # Convert back to 3 channels

        images = gaussian_applied * blurred + solarize_applied * solarized + grayscale_applied * gray
        
        # Convert to float in range [0, 1]
        images = images / 255.0
        
        return images, labels

def create_hdf5_dataset_with_dali(hdf5_path, image_dir, metadata_df, batch_size=128, num_threads=4):
    num_samples = len(metadata_df)
    img_shape = (3, 224, 224)
    
    # Prepare data for DALI pipeline
    data = {
        'image_dir': image_dir,
        'image_files': [f"{row['id']}.jpg" for _, row in metadata_df.iterrows()]
    }

    # Create DALI pipeline
    pipe = CustomImagePipeline(batch_size=batch_size, num_threads=num_threads, device_id=0, data=data)
    pipe.build()

    # Create HDF5 file and datasets
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        hdf5_file.create_dataset('images', shape=(num_samples,) + img_shape, dtype=np.float32, chunks=(1000,) + img_shape, compression="gzip", compression_opts=4)
        hdf5_file.create_dataset('latitude', shape=(num_samples,), dtype=np.float32, chunks=(1000,))
        hdf5_file.create_dataset('longitude', shape=(num_samples,), dtype=np.float32, chunks=(1000,))
        hdf5_file.create_dataset('quadtree_10_1000', shape=(num_samples,), dtype=np.int16, chunks=(1000,))

        pbar = tqdm(total=num_samples, desc="Processing Images")

        for batch_idx in range(0, num_samples, batch_size):
            pipe_out = pipe.run()
            images = pipe_out[0].as_cpu().as_array()
            
            end_idx = min(batch_idx + batch_size, num_samples)
            batch_size_actual = end_idx - batch_idx

            hdf5_file['images'][batch_idx:end_idx] = images[:batch_size_actual]
            hdf5_file['latitude'][batch_idx:end_idx] = metadata_df['latitude'].iloc[batch_idx:end_idx]
            hdf5_file['longitude'][batch_idx:end_idx] = metadata_df['longitude'].iloc[batch_idx:end_idx]
            hdf5_file['quadtree_10_1000'][batch_idx:end_idx] = metadata_df['quadtree_10_1000'].iloc[batch_idx:end_idx]

            pbar.update(batch_size_actual)

        pbar.close()

if __name__ == "__main__":
    dataset_base = 'datasets/OpenWorld'
    image_dir = f'{dataset_base}/images/train'
    csv_path = f'{dataset_base}/train_extras_removed.csv'
    hdf5_path = 'dataset_gpu_optimized.hdf5'

    metadata_df = pd.read_csv(csv_path)
    
    create_hdf5_dataset_with_dali(hdf5_path, image_dir, metadata_df)