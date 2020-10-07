import os
import pandas as pd

from .imagelist import ImageList
from typing import Optional, Tuple, List, Any


class Wind(ImageList):
    """Wind Dataset.

    Parameters:
        - **root** (str): Root directory of dataset

    """
    CLASSES = ['0', '1', '2', '3']
    image_list = {
        "C": "commercial.csv",
        "R": "residential.csv",
        "val": "validation_v2.csv",
        "test": "test_v3.csv",
        "laura": 'laura_commercial.csv'
    }
    commercial_path = '/media/feng/storage/data/commercial/commercial_buildings/'
    residential_path = '/media/feng/storage/data/newwind/'
    laura_path = '/media/feng/storage/data/commercial/commercial_laura_images/'

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])
        super(Wind, self).__init__(root, Wind.CLASSES, data_list_file=data_list_file, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, path

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse wind label csv file to data list

        Parameters:
            - **file_name** (str): The path of data file
            - **return** (list): List of (image path, class_index) tuples
        """
        df = pd.read_csv(file_name)
        if 'residential' in file_name:
            data_dir = Wind.residential_path
        elif 'laura' in file_name:
            data_dir = Wind.laura_path
        else:
            data_dir = Wind.commercial_path
        data_list = []
        for ind, row in df.iterrows():
            target = int(row['damage'])
            path = os.path.join(data_dir, row['filename'])
            data_list.append((path, target))
        return data_list
