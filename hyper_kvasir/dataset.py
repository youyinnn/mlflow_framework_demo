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

overall_num_classes = 23

excluded_class = [
    "ileum",
    # "barretts",
    "hemorrhoids",
    # "barretts-short-segment",
    # "ulcerative-colitis-grade-0-1",
    "ulcerative-colitis-grade-1-2",
    # "ulcerative-colitis-grade-2-3",
]

num_classes = overall_num_classes - len(excluded_class)
if "HyperKvasirDataModule" in sys.argv:
    if sys.argv[0] != "ae_main.py":
        sys.argv.append("--model.output_features")
        # sys.argv.append("23")
        sys.argv.append(str(num_classes))

original_dir = ["hyper_kvasir", "labeled-images"]


@dataclass
class HyperKvasirDataset(XEraseDataset):

    num_classes: int = num_classes
    data_dir: list[str] = field(default_factory=lambda: original_dir)
    test_size: float = 0.2
    head_classes_idx: tuple = (19, 6, 4, 11, 5, 10, 3, 2, 13)
    medium_classes_idx: tuple = (7, 12, 16, 9, 18, 15, 8)
    tail_classes_idx: tuple = (17, 14, 0, 1)

    def __check_if_downloaded__(self):
        return os.path.exists(os.path.join(self.data_dir, "image-labels.csv"))

    def __download_data__(self):
        print(self.data_dir)
        download_and_extract(
            self.data_dir,
            "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip",
            os.path.join(self.data_dir, ".."),
        )

    def __check_if_saliency_map_downloaded__(self):
        return os.path.exists(
            os.path.join(self.saliency_map_data_dir, "image-labels.csv")
        )

    def __download_saliency_map_data__(self):
        os.system(
            f"kaggle datasets download junhuang96/hyper-kvasir-gag-30-06 -p {self.saliency_map_data_dir} --unzip"
        )

    def __x_y_pair_list__(self) -> np.ndarray | tuple:
        label_csv = pd.read_csv(os.path.join(
            self.data_dir, "image-labels.csv"))
        img_paths = glob.glob(self.data_dir + "/**/*.jpg", recursive=True)
        label_name_and_number_map = {}
        c = 0
        for label_name in sorted(set(label_csv["Finding"].to_list())):
            if label_name.lower() not in excluded_class:
                label_name_and_number_map[label_name.lower()] = c
                c += 1
        return np.array(
            [
                [p, label_name_and_number_map[p.split("/")[-2].lower()]]
                for p in img_paths
                if not p.split("/")[-2].lower() in excluded_class
            ]
        )

    def __sal_path_transfer__(self, img_path) -> str:
        return img_path.replace(".jpg", ".png")


@dataclass
class HyperKvasirDataModule(NewBasicDataModule):

    dataset_class: Type = HyperKvasirDataset
