from backend_central_dev.data_processing.mix.aug_base import *
import os
import json
from backend_central_dev.data_processing.dataset_utils import (
    XEraseDataset,
    NewBasicDataModule,
    dataclass,
    field,
    Type,
    download_and_extract,
)
import sys
import glob
import numpy as np
import pandas as pd

overall_num_classes = 10
original_dir = ["cifar"]
class_label_map = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}


@dataclass
class CIFAR_10_PNG(XEraseDataset):

    num_classes: int = overall_num_classes
    data_dir: list[str] = field(default_factory=lambda: original_dir)

    class_labels: tuple = field(
        default_factory=lambda: list(class_label_map.keys()))

    def __check_if_downloaded__(self):
        return os.path.exists(
            os.path.join(self.data_dir, "train", "airplane", "0001.png")
        )

    def __download_data__(self):
        os.system(
            f"kaggle datasets download swaroopkml/cifar10-pngs-in-folders -p {self.data_dir} --unzip"
        )

    def __check_if_saliency_map_downloaded__(self):
        return False

    def __x_y_pair_list__(self) -> np.ndarray:
        train_path_labels = []
        test_path_labels = []
        img_paths = glob.glob(self.data_dir + "/**/*.png", recursive=True)

        for p in img_paths:
            sp = p.split('/')
            c = sp[-2]
            if "train" in p:
                train_path_labels.append([p, class_label_map[c]])
            else:
                test_path_labels.append([p, class_label_map[c]])

        return np.array(train_path_labels), np.array(test_path_labels)

    def __download_saliency_map_data__(self):
        pass

    def __sal_path_transfer__(self, img_path) -> str:
        return img_path


@dataclass
class CIFAR_10_PNG_NewDataModule(NewBasicDataModule):

    dataset_class: Type = CIFAR_10_PNG
