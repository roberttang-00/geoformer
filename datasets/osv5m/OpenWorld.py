import os
import pandas as pd
import datasets
from datasets import load_dataset

class OpenWorld(datasets.GeneratorBasedBuilder):
    def __init__(self, *args, **kwargs):
        self.full = kwargs.pop('full', False)
        super().__init__(*args, **kwargs)
        print('OpenWorld', self.__dict__)

    def _info(self):
        if self.full:
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "image": datasets.Image(),
                        "latitude": datasets.Value("float32"),
                        "longitude": datasets.Value("float32"),
                        "thumb_original_url": datasets.Value("string"),
                        "country": datasets.Value("string"),
                        "sequence": datasets.Value("string"),
                        "captured_at": datasets.Value("string"),
                        "lon_bin": datasets.Value("float32"),
                        "lat_bin": datasets.Value("float32"),
                        "cell": datasets.Value("string"),
                        "region": datasets.Value("string"),
                        "sub-region": datasets.Value("string"),
                        "city": datasets.Value("string"),
                        "land_cover": datasets.Value("float32"),
                        "road_index": datasets.Value("float32"),
                        "drive_side": datasets.Value("float32"),
                        "climate": datasets.Value("float32"),
                        "soil": datasets.Value("float32"),
                        "dist_sea": datasets.Value("float32"),
                        "quadtree_10_5000": datasets.Value("int32"),
                        "quadtree_10_25000": datasets.Value("int32"),
                        "quadtree_10_1000": datasets.Value("int32"),
                        "quadtree_10_50000": datasets.Value("int32"),
                        "quadtree_10_12500": datasets.Value("int32"),
                        "quadtree_10_500": datasets.Value("int32"),
                        "quadtree_10_2500": datasets.Value("int32"),
                        "unique_region": datasets.Value("string"),
                        "unique_sub-region": datasets.Value("string"),
                        "unique_city": datasets.Value("string"),
                        "unique_country": datasets.Value("string"),
                        "creator_username": datasets.Value("string"),
                        "creator_id": datasets.Value("string"),
                    }
                )
            )
        else:
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "image": datasets.Image(),
                        "latitude": datasets.Value("float32"),
                        "longitude": datasets.Value("float32"),
                        "lat_bin": datasets.Value("float32"),
                        "lon_bin": datasets.Value("float32"),
                        "land_cover": datasets.Value("float32"),
                        "road_index": datasets.Value("float32"),
                        "drive_side": datasets.Value("float32"),
                        "climate": datasets.Value("float32"),
                        "soil": datasets.Value("float32"),
                        "dist_sea": datasets.Value("float32"),
                        "quadtree_10_1000": datasets.Value("int32"),
                    }
                )
            )

    def df(self, annotation_path):
        if not hasattr(self, 'df_'):
            self.df_ = {}
        if annotation_path not in self.df_:
            df = pd.read_csv(annotation_path, dtype={
                'id': str, 'creator_id': str, 'creator_username': str, 
                'unique_country': str, 'unique_city': str, 'unique_sub-region': str, 'unique_region': str,
                'quadtree_10_2500': int, 'quadtree_10_500': int, 'quadtree_10_12500': int, 'quadtree_10_50000': int, 'quadtree_10_1000': int, 'quadtree_10_25000': int, 'quadtree_10_5000': int,
                'dist_sea': float, 'soil': float, 'climate': float, 'drive_side': float, 'road_index': float, 'land_cover': float, 'city': str, 'sub-region': str, 'region': str, 'cell': str, 'lat_bin': float, 'lon_bin': float, 'captured_at': str, 'sequence': str, 'country': str, 'thumb_original_url': str, 'longitude': float, 'latitude': float
            })
            if not self.full:
                df = df[['id', 'latitude', 'longitude', 'lat_bin', 'lon_bin', 'land_cover', 'road_index', 'drive_side', 'climate', 'soil', 'dist_sea', 'quadtree_10_1000']]

            df = df.set_index('id')
            self.df_[annotation_path] = df.to_dict('index')
        return self.df_[annotation_path]
    
    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_dir": self.base_path + "/images/train/",
                    "annotation_path": self.base_path + "/train_filtered.csv",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "image_dir": self.base_path + "/images/test/",
                    "annotation_path": self.base_path + "/test_filtered.csv",
                },
            ),
        ]

    def _generate_examples(self, image_dir, annotation_path):
        """Generate examples."""
        df = self.df(annotation_path)
        for idx, image_name in enumerate(os.listdir(image_dir)):
            image_path = os.path.join(image_dir, image_name)
            info_id = os.path.splitext(image_name)[0]
            try:
                example = {
                    "image": image_path,
                } | df[info_id]
            except Exception as e:
                print(f'Exception {e} for id: {info_id}, idx: {idx}, image_path: {image_path}')
                continue
            yield idx, example

if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("./OpenWorld.py")
    print(dataset)