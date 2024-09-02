import lmdb
import os
import csv
import json

MAP_SIZE = 30 * 1024 * 1024 * 1024

def create_lmdb(images_dir, csv_path, lmdb_path):
    env = lmdb.open(lmdb_path, map_size=MAP_SIZE)

    # Read CSV file
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        metadata = {row['id']: row for row in reader}

    with env.begin(write=True) as txn:
        idx = 0
        for root, _, files in os.walk(images_dir):
            for filename in files:
                if filename.lower().endswith('.jpg'):
                    image_path = os.path.join(root, filename)
                    
                    filename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
                    if filename_no_ext in metadata:
                        with open(image_path, 'rb') as f:
                            image_bytes = f.read()
                        
                        # Store image
                        txn.put(f'image_{idx}'.encode(), image_bytes)
                        
                        # Store metadata
                        meta_key = f'meta_{idx}'.encode()
                        meta_value = json.dumps(metadata[filename_no_ext]).encode()
                        txn.put(meta_key, meta_value)
                        
                        idx += 1
                        if idx % 1000 == 0:
                            print(f'Processed {idx} images')
        txn.put('length'.encode(), str(idx).encode())
    
    env.close()
    print(f'Finished creating LMDB database with {idx} entries')
    
base_path = 'datasets/osv5m'
create_lmdb(f'{base_path}/images/train', f'{base_path}/train_extras_removed.csv', f'{base_path}/images/train/train_db')