# autopep8: off
import os
import sys
from backend_central_dev.constant import TaskStatus
from backend_central_dev.xai.xai_eval_task import task_wrapper
from backend_central_dev.model_training.lightning_model import MyLightningModel
from backend_central_dev.data_processing.dataset_utils import NewBasicDataModule
from backend_central_dev.utils.pytorch_utils import get_device
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lightning.pytorch.loggers import CSVLogger
from rcap import *
# autopep8: on


def rcap_func(model: MyLightningModel, datamodule: NewBasicDataModule, xai_modules):

    datamodule.batch_size = 32
    datamodule.setup('val')
    td = datamodule.test_dataloader()
    device = get_device()
    print("Using device:", device)
    for images, targets in td:
        images, targets = images.to(device), targets.to(device)
        rs = batch_rcap(
            model.to(device), (images, targets),
            xai_modules['gradient_methods'].guided_absolute_grad,
            {}
        )['overall_rcap']['RCAP']
        break
    return {
        'rcap': rs.mean()
    }


def _xai_eval(task_ticket, publisher_endpoint_url, task_parameters):
    print(task_parameters)
    task_wrapper(task_ticket, publisher_endpoint_url,
                 task_parameters, rcap_func)
    # get model, dataset,
    return TaskStatus.finished
