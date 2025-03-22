import torch.nn as nn
from torchvision import models
from backend_central_dev.model_training.lightning_model import FineTunableModel


def model_init_func():
    return models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )


def model_fine_tune_func(model, output_features):
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(
        in_features=in_features, out_features=output_features
    )
    return model
