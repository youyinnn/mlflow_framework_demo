from backend_central_dev.data_processing.mix.aug_base import *
from backend_central_dev.data_processing.dataset_utils import (
    XEraseDataset,
    NewBasicDataModule,
    dataclass,
    field,
    Type,
)
import numpy as np


@dataclass
class DummyDataset(XEraseDataset):

    num_classes: int = 10
    data_dir: list[str] = field(default_factory=lambda: [])
    test_size: float = 0.2

    def __check_if_downloaded__(self):
        return True

    def __x_y_pair_list__(self) -> np.ndarray | tuple:
        return None


@dataclass
class DummyDataModule(NewBasicDataModule):

    dataset_class: Type = DummyDataset
