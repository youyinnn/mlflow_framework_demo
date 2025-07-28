import os
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_fn.to(x.device)(output, y)
        self.eval_metrics("train", batch, batch_idx, output, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_fn.to(x.device)(output, y)
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
        mlflow.log_metrics(metrics, step=self.trainer.global_step)

if __name__ == "__main__":
    L.seed_everything(42)
    mlflow.set_tracking_uri('http://localhost:15001')
    # device = "mps"
    device = "cuda"
    batch_size = 256
    output_features = 10

    user_home = os.path.expanduser('~')
    data_root = os.path.join(user_home, 'autodl-tmp',
                            'ml_data', 'cifar', 'cifar10', 'cifar10')

    trainsform = transforms.Compose([
        transforms.Resize((32, 32)),
        # transforms.ToTensor(),
    ])
    
    print("Transform:", trainsform)

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
    
    
    with mlflow.start_run():
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

        print("Start train")
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

        ckpt_path = trainer.checkpoint_callback.best_model_path
        
        # ckpt_path = "/root/mlflow_framework_demo/lightning_logs/version_2/checkpoints/best-epoch=39-val_acc=0.8428.ckpt"
        
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

        print("Skip XAI")

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