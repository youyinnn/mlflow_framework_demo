import json
import multiprocessing
import os
import sys
import time
import traceback
import mlflow
import numpy as np
import torch
from torchinfo import summary
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
import lightning as L
from mlflow_demo.xai_eval import rcap
from mlflow_demo.xai import gradient_methods
from tqdm import tqdm
from glob import glob

from torchvision import models
import torchmetrics

from backend_central_dev.utils import data_utils

class MyImageDataset(Dataset):
    def __init__(self, img_dir, train=True, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

        if self.train:
            self.image_paths = glob(self.img_dir + "/train/**/*.png", recursive=True)
        else:
            self.image_paths = glob(self.img_dir + "/test/**/*.png", recursive=True)
        
        self.labels = [os.path.basename(os.path.dirname(path)) for path in self.image_paths]
        
        self.class_label_map = {
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
        
        print(f"Found {len(self.image_paths)} images in {self.img_dir} for {'train' if self.train else 'test'} set.")
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.class_label_map[self.labels[idx]]
        image = data_utils.read_image_to_tensor_from_path(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

class MyLightModel(L.LightningModule):
    def __init__(
        self,
        output_features,
        loss_fn_key="CrossEntropyLoss",
        loss_fn_hparams: dict = {},
        optimizer_key="Adam",
        optimizer_hparams: dict = {},
        use_mlflow=True,
        task="multiclass",
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[])

        model = models.efficientnet_v2_s(
            weights=weights
        )
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(
            in_features=in_features, out_features=output_features
        )
        self.model = model

        self.loss_fn = getattr(torch.nn, loss_fn_key)(**loss_fn_hparams)

        self.metrics = dict(
            acc=torchmetrics.Accuracy(
                task=task, top_k=1, num_classes=output_features),
            macro_acc=torchmetrics.Accuracy(
                task=task, top_k=1, average="macro", num_classes=output_features
            ),
            f1=torchmetrics.F1Score(
                task=task, average="macro", num_classes=output_features),
            roc_auc=torchmetrics.AUROC(task=task, num_classes=output_features),
        )

        self.st = None
        self.y_true = np.array([])
        self.y_pred = np.array([])
        self.losses = np.array([])
        self.cm = None
        self.cr = None
        self.metrics_in_batch = {}

    def forward(self, x):
        return self.model(x)
    
    def get_loss(self, x, y, batch_idx):
        if hasattr(self.model, "get_output_loss"):
            return self.model.get_output_loss(x, y, self.loss_fn)
        else:
            output = self.model(x)
            loss = self.loss_fn(output, y)
            return output, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, loss = self.get_loss(x, y, batch_idx)
        self.eval_metrics("train", batch, batch_idx, output, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output, loss = self.get_loss(x, y, batch_idx)
        self.eval_metrics("val", batch, batch_idx, output, loss)

    def configure_optimizers(self):
        params = self.model.parameters()

        self.optimizer = getattr(torch.optim, self.hparams.optimizer_key)(
            params, **self.hparams.optimizer_hparams
        )
        return self.optimizer

    def eval_metrics(self, event_key, batch, batch_idx, output, loss):
        x, y = batch
        metrics = {}
        if self.hparams.task == "multiclass":
            _y = y.detach().cpu()
            odc = output.detach().cpu()
            for k, v in self.metrics.items():
                key = f"{event_key}_{k}"

                # we have to make the y to cpu to avoid MPS glitch
                if len(_y.shape) > 1:
                    _y = _y.argmax(dim=1)
                score = v(odc.float(), _y).item()
                metrics[key] = score
                # self.trainer.progress_bar_metrics[k] = metrics[key]
                if event_key == "val":
                    if self.metrics_in_batch.get(key) is None:
                        self.metrics_in_batch[key] = []
                    self.metrics_in_batch[key].append(score)

            if event_key == "val":
                self.y_true = np.append(self.y_true, _y.numpy())
                self.y_pred = np.append(
                    self.y_pred, torch.argmax(odc, dim=1).numpy()
                )
                self.losses = np.append(self.losses, loss.cpu().item())

                ytp = torch.from_numpy(self.y_pred)
                ytt = torch.from_numpy(self.y_true)
                overall_val_acc = self.metrics["acc"](ytp, ytt)
                self.trainer.progress_bar_metrics["max_val_acc"] = max(
                    overall_val_acc,
                    self.trainer.progress_bar_metrics.get("max_val_acc", 0.0),
                )
                

        metrics[f"{event_key}_loss"] = loss.item()
        self.trainer.progress_bar_metrics["loss"] = metrics[f"{event_key}_loss"]
        self.log_dict(metrics)
        if self.hparams.use_mlflow:
            mlflow.log_metrics(metrics, step=self.trainer.global_step)


def task_execution_wrapper(
    run_name,
    task_function,
    task_parameters
):
    p = multiprocessing.current_process()
    task_execution_info = dict(
        task_parameters=task_parameters,
        run_name=run_name,
        pid=p.pid
    )
    print(json.dumps(task_execution_info, indent=4))

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    ticket_log_dir = os.path.join("logs", run_name)
    os.makedirs(ticket_log_dir, exist_ok=True)

    std_out_file_path = os.path.join(
        ticket_log_dir, f"out.log")
    std_err_file_path = os.path.join(
        ticket_log_dir, f"err.log")
    std_out_file = open(std_out_file_path, 'w')
    std_err_file = open(std_err_file_path, 'w')

    try:
        sys.stdout = std_out_file
        sys.stderr = std_err_file
        task_function(run_name, task_parameters)
        
    except Exception as e:
        print(''.join(
            traceback.TracebackException.from_exception(e).format()),
            file=sys.stderr)
        # raise e
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        std_out_file.close()
        std_err_file.close()
        with open(std_out_file_path) as f:
            std_output_str = f.readlines()
        with open(std_err_file_path) as f:
            std_error_str = f.readlines()

def task_function_train(run_name, task_parameters):
    L.seed_everything(42)
    mlflow.set_tracking_uri('http://localhost:15001')

    batch_size = task_parameters.get('batch_size', 256)
    device = task_parameters.get('device', 'cuda')
    
    trainsform = transforms.Compose([
        transforms.Resize((32, 32)),
        # transforms.ToTensor(),
    ])
    
    print("Transform:", trainsform)

    user_home = os.path.expanduser('~')
    data_root = os.path.join(user_home, 'autodl-tmp',
                            'ml_data', 'cifar', 'cifar10', 'cifar10')
    
    train_dataset = MyImageDataset(
        data_root, train=True, transform=trainsform)
    test_dataset = MyImageDataset(
        data_root, train=False, transform=trainsform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)


    model = MyLightModel(
        10,
        optimizer_hparams=dict(
            lr=0.0001,
            weight_decay=4.0e-05
        )
    )
    
    trainer_params = dict(
        # max_epochs=5,
        max_epochs=40,
        # max_epochs=1,
        # limit_train_batches=1,
        # limit_test_batches=1,
        # limit_val_batches=1,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor='val_acc',
                save_top_k=1,
                filename='best-{epoch}-{val_acc:.4f}',
                mode='max',
                save_weights_only=True
            )
        ],
    )
    
    trainer = L.Trainer(
        precision='16-mixed',
        **trainer_params
    )
    
    
    with mlflow.start_run(run_name=run_name):
        L.seed_everything(42)
        run = mlflow.active_run()
        # Log training parameters.
        mlflow.log_params(model.hparams)
        mlflow.log_params({'trainer_params': trainer_params})

        # Log model summary.
        artifact_root_path = os.path.join("runs", run.info.run_id)
        model_summary_path = os.path.join(artifact_root_path, "model_summary.txt")
        os.makedirs(os.path.dirname(model_summary_path), exist_ok=True)
        with open(model_summary_path, "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact(model_summary_path)

        st = time.time()
        print("Start train")
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
        print("Train finished in", time.time() - st, "seconds")

        st = time.time()
        ckpt_path = trainer.checkpoint_callback.best_model_path
        # ckpt_path = "/root/mlflow_framework_demo/lightning_logs/version_2/checkpoints/best-epoch=39-val_acc=0.8428.ckpt"
        # ckpt_path = "/root/mlflow_framework_demo/best-epoch=38-val_acc=0.8412.ckpt"
        
        ckpt = torch.load(
            ckpt_path, map_location=device, weights_only=False)
        
        model = MyLightModel(
            10,
            optimizer_hparams=dict(
                lr=0.0001,
                weight_decay=4.0e-05
            )
        )
        
        model.load_state_dict(ckpt['state_dict'], strict=True)

        print("Start val")
        trainer.validate(model, test_dataloader)
        print("Validation finished in", time.time() - st, "seconds")

        print("Skip XAI")

        st = time.time()
        print("XAI Eval")
        model.eval()
        all = []
        for images, targets in tqdm(test_dataloader):
            images, targets = images.to(device), targets.to(device)
            rs = rcap.batch_rcap(
                model.to(device), (images, targets),
                gradient_methods.guided_absolute_grad,
                {}
            )['overall_rcap']['RCAP']
            mlflow.log_metric('rcap', rs.mean())
            all.extend(rs)
        
        print("RCAP mean:", np.array(all).mean())
        mlflow.log_metric('rcap', np.array(all).mean())
        print("XAI Eval finished in", time.time() - st, "seconds")
        
def task_function_train_van(run_name, task_parameters):
    L.seed_everything(42)

    batch_size = task_parameters.get('batch_size', 256)
    device = task_parameters.get('device', 'cuda')
    
    trainsform = transforms.Compose([
        transforms.Resize((32, 32)),
        # transforms.ToTensor(),
    ])
    
    print("Transform:", trainsform)

    user_home = os.path.expanduser('~')
    data_root = os.path.join(user_home, 'autodl-tmp',
                            'ml_data', 'cifar', 'cifar10', 'cifar10')
    
    train_dataset = MyImageDataset(
        data_root, train=True, transform=trainsform)
    test_dataset = MyImageDataset(
        data_root, train=False, transform=trainsform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)


    model = MyLightModel(
        10,
        optimizer_hparams=dict(
            lr=0.0001,
            weight_decay=4.0e-05
        ),
        use_mlflow = False
    )
    
    trainer_params = dict(
        max_epochs=40,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor='val_acc',
                save_top_k=1,
                filename='best-{epoch}-{val_acc:.4f}',
                mode='max',
                save_weights_only=True
            )
        ],
    )
    
    trainer = L.Trainer(
        precision='16-mixed',
        **trainer_params
    )
    
    
    L.seed_everything(42)

    st = time.time()
    print("Start train")
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    print("Train finished in", time.time() - st, "seconds")

    st = time.time()
    ckpt_path = trainer.checkpoint_callback.best_model_path
    
    # ckpt_path = "/root/mlflow_framework_demo/lightning_logs/version_2/checkpoints/best-epoch=39-val_acc=0.8428.ckpt"
    # ckpt_path = "/root/mlflow_framework_demo/best-epoch=38-val_acc=0.8412.ckpt"
    
    ckpt = torch.load(
        ckpt_path, map_location=device, weights_only=False)
    
    model = MyLightModel(
        10,
        optimizer_hparams=dict(
            lr=0.0001,
            weight_decay=4.0e-05
        ),
        use_mlflow = False
    )
    
    model.load_state_dict(ckpt['state_dict'], strict=True)

    print("Start val")
    trainer.validate(model, test_dataloader)
    print("Validation finished in", time.time() - st, "seconds")

    print("Skip XAI")

    st = time.time()
    print("XAI Eval")
    model.eval()
    all = []
    for images, targets in tqdm(test_dataloader):
        images, targets = images.to(device), targets.to(device)
        rs = rcap.batch_rcap(
            model.to(device), (images, targets),
            gradient_methods.guided_absolute_grad,
            {}
        )['overall_rcap']['RCAP']
        print("RCAP mean:", rs.mean())
        all.extend(rs)
    
    print("Final RCAP mean:", np.array(all).mean())
    print("XAI Eval finished in", time.time() - st, "seconds")

def task_function_train_mlxops(run_name, task_parameters):
    L.seed_everything(42)
    # from cifar_dataset import CIFAR_10_PNG_NewDataModule
    # from backend_central_dev.model_training.lightning_model import MlxOps_LightningModel, FineTunableModel
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
    
    # dm = CIFAR_10_PNG_NewDataModule(
    #     img_size=32,
    #     batch_size=task_parameters.get('batch_size', 256),
    # )
    
    trainsform = transforms.Compose([
        transforms.Resize((32, 32)),
        # transforms.ToTensor(),
    ])
    
    print("Transform:", trainsform)

    user_home = os.path.expanduser('~')
    data_root = os.path.join(user_home, 'autodl-tmp',
                            'ml_data', 'cifar', 'cifar10', 'cifar10')
    
    train_dataset = MyImageDataset(
        data_root, train=True, transform=trainsform)
    test_dataset = MyImageDataset(
        data_root, train=False, transform=trainsform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=task_parameters.get('batch_size', 256),
                                shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=task_parameters.get('batch_size', 256),
                                shuffle=False, num_workers=0)
    
    # model = MlxOps_LightningModel(
    #     FineTunableModel(model_init_func=model_init_func, model_fine_tune_func=model_fine_tune_func),
    #     output_features=10,
    #     optimizer_hparams=dict(
    #         lr=0.0001,
    #         weight_decay=4.0e-05
    #     )
    # )
    
    model = MyLightModel(
        10,
        optimizer_hparams=dict(
            lr=0.0001,
            weight_decay=4.0e-05
        )
    )
    
    
    trainer_params = dict(
        max_epochs=5,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor='val_acc',
                save_top_k=1,
                filename='best-{epoch}-{val_acc:.4f}',
                mode='max',
                save_weights_only=True
            )
        ],
    )
    
    trainer = L.Trainer(
        precision='16-mixed',
        **trainer_params
    )
    
    print("Start train")
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn', force=True)
    process = multiprocessing.Pool(processes=1)
    
    args = [
        # "cifar10-mlflow-demo" + str(time.time()),
        "cifar10-mlflow-demo-van-time" + str(time.time()),
        # task_function_train,
        task_function_train_van,
        # task_function_train_mlxops,
        dict(
            batch_size=256,
            device='cuda'
        )
    ]
    as_rs = process.apply_async(
        task_execution_wrapper,
        args=args
    )
    
    as_rs.wait()