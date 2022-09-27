from PIL import Image
import pytorch_lightning as pl
from typing import Tuple, List
from glob import glob
import numpy as np
import torch
from torch import nn
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import common_functions
import os
from tqdm import tqdm


class DMLModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            loss_func: nn.Module,
            lr: float=1e-6,
            lr_loss: float=1e-6,
            weight_decay: float=0.0001,
            momentum: float=0.9,
            train_transform: torch.nn.Module=None,
            additionally_corrupt_background_during_val_and_test: bool=True,
            train_with_corrupted_background: bool=False,
        ) -> None:
        super().__init__()

        self.model = model
        self.loss_func = loss_func
        self.lr = lr
        self.lr_loss = lr_loss
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.transform = train_transform
        self.additionally_corrupt_background_during_val_and_test = additionally_corrupt_background_during_val_and_test
        self.train_with_corrupted_background = train_with_corrupted_background

        if self.additionally_corrupt_background_during_val_and_test or self.train_with_corrupted_background:
            directory = os.path.dirname(os.path.realpath(__file__))
            self.val_background_images = [Image.open(i) for i in tqdm(glob(directory + f"/../../data/unsplash_backgrounds/val/*.jpeg"))]
            self.test_background_images = [Image.open(i) for i in tqdm(glob(directory + f"/../../data/unsplash_backgrounds/test/*.jpeg"))]
            assert len(self.val_background_images) == 100
            assert len(self.test_background_images) == 100

        self.accuracy_calculator = AccuracyCalculator(
            device=torch.device("cpu"),
            k="max_bin_count",
            # We only use these metrics since they can be computed with only max_bin_count as k,
            # which should fit into memory and be faster to compute
            include=["mean_average_precision_at_r", "precision_at_1", "r_precision"]
        )
        # we ignore train_transform because else it won't work
        self.save_hyperparameters(ignore=["train_transform"])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        self.model.apply(common_functions.set_layers_to_eval("BatchNorm"))

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        inputs = batch["inputs"]
        # To differentiate the input, we need to require gradients
        inputs.requires_grad_(True)

        labels = batch["labels"]
        masks = batch["masks"]

        if self.train_with_corrupted_background:
            bg_indices = np.random.choice(len(self.val_background_images), size=len(masks), replace=True)
            bg_images = [self.val_background_images[i] for i in bg_indices]

            bg_images = [self.transform(i, i.split()[0])[0].to(inputs.device) for i in bg_images]
            bg_images = torch.stack(bg_images)

            inputs = inputs * masks + bg_images * (1 - masks)

        # handles the forward pass as well
        loss = self.loss_func(self.model, inputs, labels, masks)
    
        for key, item in loss.items():
            self.log(f"train_{key}", item)
        loss = loss["loss"]

        return loss
    
    def val_test_step(self, batch: torch.Tensor, batch_idx: int, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["inputs"]
        y = batch["labels"]
        embedding = self.model(x).detach()

        if self.additionally_corrupt_background_during_val_and_test:
            masks = batch["masks"]

            bg_indices = np.random.choice(len(self.val_background_images), size=len(masks), replace=True)

            # corrupt the background images
            if mode == "val":
                bg_images = [self.val_background_images[i] for i in bg_indices]
            elif mode == "test":
                bg_images = [self.test_background_images[i] for i in bg_indices]
            else:
                raise ValueError(f"mode must be either val or test, but is {mode}")

            bg_images = [self.transform(i, i.split()[0])[0].to(x.device) for i in bg_images]
            bg_images = torch.stack(bg_images)

            x_corrupted = x * masks + bg_images * (1 - masks)
            embedding_corrupted = self.model(x_corrupted).detach()

            return embedding, y, embedding_corrupted

        return embedding, y

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.val_test_step(batch, batch_idx, mode="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.val_test_step(batch, batch_idx, mode="test")

    def val_test_epoch_end(self, outputs: List[Tuple[torch.Tensor, torch.Tensor]], mode: str) -> None:
        embeddings = torch.cat([o[0] for o in outputs], dim=0).cpu().numpy()
        all_labels = torch.cat([o[1] for o in outputs], dim=0).cpu().numpy()

        # save test embeddings as pickle file to better analyze resulting embeddings
        # if mode == "test":
        #     torch.save(
        #             {"embeddings": embeddings, "labels": all_labels},
        #             self.logger.log_dir + "/test_embeddings.p"
        #         )

        acc = self.accuracy_calculator.get_accuracy(embeddings, embeddings, all_labels, all_labels, embeddings_come_from_same_source=True)
        
        # log metrics
        for key, item in acc.items():
            name = f"{mode}_{key}"
            self.log(name, item)

        # also validate and test the model using the corrupted data
        if self.additionally_corrupt_background_during_val_and_test:
            embeddings = torch.cat([o[2] for o in outputs], dim=0).cpu().numpy()
            acc = self.accuracy_calculator.get_accuracy(embeddings, embeddings, all_labels, all_labels, embeddings_come_from_same_source=True)
            for key, item in acc.items():
                name = f"{mode}_corrupted_{key}"
                self.log(name, item)

    def validation_epoch_end(self, outputs: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.val_test_epoch_end(outputs, mode="val")

    def test_epoch_end(self, outputs: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.val_test_epoch_end(outputs, mode="test")

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
                            [
                                {"params": self.model.parameters(), "lr": self.lr},
                                {"params": self.loss_func.parameters(), "lr": self.lr_loss}
                            ],
                            weight_decay=self.weight_decay,
                            momentum=self.momentum
                        )

        return optimizer
