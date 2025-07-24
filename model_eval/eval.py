from datetime import datetime, timedelta
import os
import time

from backend_central_dev.constant import *
from backend_central_dev.xai_sdk import (
    get_trained_model_and_dataset_from_services,
    get_trainer_configuration
)
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import CSVLogger
import torch
import torchmetrics

from torch_uncertainty.metrics.classification import BrierScore

from backend_central_dev.utils.pytorch_utils import get_device
from backend_central_dev.utils.loggers import *
from tqdm import tqdm
import pandas as pd


def get_brier_score(model, x, y, brier_score, num_classes):
    pred = model(x)
    softmax_pred = torch.softmax(pred, dim=1)
    # return None, pred, softmax_pred
    y = torch.nn.functional.one_hot(
        y, num_classes=num_classes).float()
    brier_score.update(softmax_pred.detach(), y.detach())

    return brier_score, pred, softmax_pred


def brier_score_eval(task_ticket, publisher_endpoint_url, task_parameters):
    L.seed_everything(42)

    model, datamodule = get_trained_model_and_dataset_from_services(
        publisher_endpoint_url,
        task_parameters[TaskExecution.previous_task_ticket],
        task_parameters
    )

    brier_score = BrierScore(
        num_classes=datamodule.num_classes
    ).to(get_device())

    datamodule.setup('val')
    model.eval()
    model.to(get_device())
    val_dataloader = datamodule.val_dataloader()
    c = 0
    rs = []
    for batch in tqdm(val_dataloader, desc="Evaluating Brier Score"):
        c += 1
        x, y = batch
        x, y = x.to(get_device()), y.to(get_device())
        bs, pred, softmax_pred = get_brier_score(
            model, x, y, brier_score, datamodule.num_classes)
        rs.append({
            "brier_score": bs,
            "batch": c
        })
        print_to_log(f"Brier Score: {bs}")
        if c > 10:
            break

    df = pd.DataFrame(rs)
    csv_dir = os.path.join(
        os.environ['COMPONENT_STATIC_PATH'],
        "rs",
        task_ticket
    )
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(
        os.path.join(csv_dir, "metrics.csv"),
        index=False
    )


def vanilla_eval(task_ticket, publisher_endpoint_url, task_parameters):
    L.seed_everything(42)

    model, datamodule = get_trained_model_and_dataset_from_services(
        publisher_endpoint_url,
        task_parameters[TaskExecution.previous_task_ticket],
        task_parameters
    )

    trainer_cfg, _, no_train = get_trainer_configuration(
        publisher_endpoint_url,
        task_parameters[Configuration.trainer_configuration_id]
    )

    # model evaluation as csv
    trainer_cfg['logger'] = CSVLogger(
        os.environ['COMPONENT_STATIC_PATH'],
        name="rs", version=task_ticket
    )

    trainer = L.Trainer(
        precision='16-mixed',
        # fast_dev_run=True,
        **trainer_cfg
    )

    trainer.callbacks = [
        RichProgressBar(),
        *trainer.callbacks,
    ]
    # validate the model with lightning steps
    trainer.validate(model, datamodule=datamodule)


def macro_acc_eval(task_ticket, publisher_endpoint_url, task_parameters):
    L.seed_everything(42)
    model, datamodule = get_trained_model_and_dataset_from_services(
        publisher_endpoint_url,
        task_parameters[TaskExecution.previous_task_ticket],
        task_parameters
    )

    model.metrics['macro_acc'] = torchmetrics.Accuracy(
        task="multiclass", top_k=1, average="macro",
        num_classes=datamodule.num_classes
    )

    trainer_cfg, _, no_train = get_trainer_configuration(
        publisher_endpoint_url,
        task_parameters[Configuration.trainer_configuration_id]
    )

    # model evaluation as csv
    trainer_cfg['logger'] = CSVLogger(
        os.environ['COMPONENT_STATIC_PATH'],
        name="rs", version=task_ticket
    )

    trainer = L.Trainer(
        precision='16-mixed',
        limit_val_batches=40,
        **trainer_cfg
    )

    trainer.callbacks = [
        RichProgressBar(),
        *trainer.callbacks,
    ]
    # validate the model with lightning steps
    trainer.validate(model, datamodule=datamodule)


def basic_score_eval(task_ticket, publisher_endpoint_url, task_parameters):
    L.seed_everything(42)
    model, datamodule = get_trained_model_and_dataset_from_services(
        publisher_endpoint_url,
        task_parameters[TaskExecution.previous_task_ticket],
        task_parameters
    )

    model.metrics['macro_acc'] = torchmetrics.Accuracy(
        task="multiclass", top_k=1, average="macro",
        num_classes=datamodule.num_classes
    )

    model.metrics['macro_f1'] = torchmetrics.F1Score(
        task="multiclass", top_k=1, average="macro",
        num_classes=datamodule.num_classes
    )

    trainer_cfg, _, no_train = get_trainer_configuration(
        publisher_endpoint_url,
        task_parameters[Configuration.trainer_configuration_id]
    )

    # model evaluation as csv
    trainer_cfg['logger'] = CSVLogger(
        os.environ['COMPONENT_STATIC_PATH'],
        name="rs", version=task_ticket
    )

    trainer = L.Trainer(
        precision='16-mixed',
        **trainer_cfg
    )

    trainer.callbacks = [
        RichProgressBar(),
        *trainer.callbacks,
    ]
    # validate the model with lightning steps
    trainer.validate(model, datamodule=datamodule)
