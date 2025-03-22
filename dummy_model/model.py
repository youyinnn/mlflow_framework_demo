

import pathlib
import os
from types import SimpleNamespace
from zipfile import ZipFile
import torch.nn as nn
import geffnet
import torch
from backend_central_dev.utils.pytorch_utils import get_device


def model_init_func(**model_init_kwargs):
    print("dummy_model/model.py: model_init_func")
    return


def model_fine_tune_func(model, output_features):
    print("dummy_model/model.py: model_fine_tune_func")
    return
