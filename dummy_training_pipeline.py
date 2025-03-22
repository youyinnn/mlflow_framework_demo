import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from dummy_dataset.dataset import DummyDataModule
from backend_central_dev.xai_sdk import get_lightning_model
import lightning as L
import mlflow
import multiprocessing

mlflow.set_tracking_uri('http://localhost:8080')

# Auto log all MLflow entities
mlflow.pytorch.autolog()


def dummy_train():
    # train the model
    dm = DummyDataModule(
        img_size=128,
        batch_size=32,
    )
    lightning_model = get_lightning_model(
        {},
        dm,
        "dummy_model/model.py"
    )

    trainer = L.Trainer(
        precision='16-mixed',
        # fast_dev_run=True,
        **{}
    )


dummy_eval = dummy_train


def run_with_multiprocessing(func):
    with multiprocessing.Pool(processes=1) as pool:
        pool.apply(func)


if __name__ == "__main__":
    rs_data = []
    num_exp = 100
    for _ in tqdm(range(num_exp)):
        with mlflow.start_run() as run:
            st1 = time.time()
            run_with_multiprocessing(dummy_train)
            et1 = time.time()

            st2 = time.time()
            run_with_multiprocessing(dummy_eval)
            et2 = time.time()

        rs_data.append((et1 - st1, et2 - st2))

    rs_data = np.array(rs_data)
    # print(rs_data)
    print(rs_data.mean(axis=0))
