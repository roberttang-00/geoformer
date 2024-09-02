import torch
from torch.utils.data import Dataset, DataLoader
import time
import lmdb
import numpy as np
import os

def test_performance(dataset, batch_size=32, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    start_time = time.time()
    for _ in loader:
        pass
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    image_dir = "path/to/your/images"
    lmdb_path = "path/to/your/lmdb"
    mmap_file = "path/to/your/mmap.npy"
    tensor_file = "path/to/your/tensors.pt"
    tar_file = "path/to/your/dataset.tar"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Test Base Dataset
    base_dataset = BaseDataset(image_dir, transform)
    base_time = test_performance(base_dataset)
    print(f"Base Dataset Time: {base_time:.2f} seconds")

    # Test LMDB Dataset
    lmdb_dataset = LMDBDataset(lmdb_path, transform)
    lmdb_time = test_performance(lmdb_dataset)
    print(f"LMDB Dataset Time: {lmdb_time:.2f} seconds")

    # Test Memory-mapped Dataset
    mmap_dataset = MemMapDataset(mmap_file, transform)
    mmap_time = test_performance(mmap_dataset)
    print(f"Memory-mapped Dataset Time: {mmap_time:.2f} seconds")

    # Test Preprocessed Tensor Dataset
    tensor_dataset = TensorDataset(tensor_file)
    tensor_time = test_performance(tensor_dataset)
    print(f"Preprocessed Tensor Dataset Time: {tensor_time:.2f} seconds")

    # Test WebDataset
    webdataset = wds.WebDataset(tar_file).decode("pil").to_tuple("jpg")
    web_time = test_performance(webdataset)
    print(f"WebDataset Time: {web_time:.2f} seconds")