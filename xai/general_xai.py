# autopep8: off
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import gradient_methods
from backend_central_dev.xai.xai_task import task_wrapper
# autopep8: on


def Guided_AbsoluteGrad_xai(task_ticket, publisher_endpoint_url, task_parameters):
    print('===========', publisher_endpoint_url)
    return task_wrapper(
        task_ticket, publisher_endpoint_url,
        task_parameters, gradient_methods.guided_absolute_grad
    )
