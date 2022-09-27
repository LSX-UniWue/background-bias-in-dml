#! /usr/bin/env python3

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import torch
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
import os
import tarfile
import logging
from tqdm import tqdm
from PIL import Image
import pytorch_lightning as pl
from typing import Union, Optional, List, Dict
from pytorch_metric_learning.samplers import MPerClassSampler
from pathlib import Path


from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.datasets.utils import check_integrity
import gdown
from gdown.cached_download import assert_md5sum
import os
import tarfile

class Cub200(Dataset):
    url = 'https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    md5 = '97eceeb196236b17998738112f37df78'

    filename_masks = 'images_masked.tgz'

    def __init__(self, root, mode="train", transform=None, download=False):
        self.root = os.path.join(root, "cub2011")
        os.makedirs(self.root, exist_ok=True)
        self.mode = mode

        assert self.mode in ["train", "val", "test", "all"]

        if download:
            try:
                self.set_paths_and_labels()
            except:
                self.download_dataset()
                self.set_paths_and_labels()
        else:
            self.set_paths_and_labels()
        
        self.transform = transform
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform is not None:
            img, mask = self.transform(img, mask)

            assert img.shape == (3, 227, 227), f"Image not in correct shape: {img.shape}"
            assert mask.shape == (1, 227, 227), f"Mask not in correct shape: {mask.shape}"

        return {
            "inputs": img,
            "labels": label,
            "masks": mask
        }
        
    def set_paths_and_labels(self):
        img_folder = os.path.join(self.root, "CUB_200_2011", "images")
        self.dataset = datasets.ImageFolder(img_folder)
        self.labels = np.array([label for (path, label) in self.dataset.imgs])
        # prepare paths to masks
        # e.g. the path
        # data/cub2011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg
        # should become
        # data/cub2011/images_masked/Black_Footed_Albatross_0001_796111.png
        self.mask_paths = [path for (path, label) in self.dataset.imgs]
        self.mask_paths = ["/".join(p.split("/")[:-4] + ["images_masked"] + p.split("/")[-1:]).replace("jpg", "png") for p in self.mask_paths]

        assert len(np.unique(self.labels)) == 200
        assert self.__len__() == 11788

        if self.mode == "train":
            # nonzero returns a tuple (one element per dimension)
            indices = np.nonzero(self.labels < 75)[0]
        elif self.mode == "val":
            indices = np.nonzero(np.logical_and((self.labels >= 75), (self.labels < 100)))[0]
        elif self.mode == "test":
            indices = np.nonzero(self.labels >= 100)[0]
        else: # all
            indices = np.arange(len(self.labels))
        
        self.dataset = Subset(self.dataset, indices)
        self.labels = self.labels[indices]
        self.mask_paths = [self.mask_paths[i] for i in indices]

        assert len(self.dataset) == len(self.labels)


    def download_dataset(self):
        output_location = os.path.join(self.root, self.filename)
        if check_integrity(output_location, self.md5):
            print('Using downloaded and verified file: ' + output_location)
        else:
            gdown.download(self.url, output_location, quiet=False)
            assert_md5sum(output_location, self.md5)
        with tarfile.open(output_location, "r:gz") as tar:
            tar.extractall(path=self.root)
        with tarfile.open(os.path.join(self.root, self.filename_masks), "r:gz") as tar:
            tar.extractall(path=self.root)


class Cub200DataModule(pl.LightningDataModule):
    def __init__(
            self,
            root: str,
            batch_size: int = 32,
            num_workers: int = 4,
            m_per_class: int = 4,
            train_transform = None,
            val_transform = None
        ):
        super().__init__(train_transforms=train_transform, val_transforms=val_transform, test_transforms=val_transform)
        self.root = root
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train_classes = 75
        self.m_per_class = m_per_class

    def prepare_data(self) -> None:
        # Download data if not present
        Cub200(self.root, download=True)

    def setup(self, stage: Optional[str]) -> None:
        self.train_data = Cub200(self.root, mode="train", transform=self.train_transform, download=False)
        self.val_data = Cub200(self.root, mode="val", transform=self.val_transform, download=False)
        self.test_data = Cub200(self.root, mode="test", transform=self.val_transform, download=False)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        # length_before_new_iter is set to 100 iterations per epoch, based on the `iterations_per_epoch` value from the powerful-benchmarker default config
        sampler = MPerClassSampler(self.train_data.labels, self.m_per_class, self.batch_size, length_before_new_iter=100*self.batch_size)
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler, drop_last=True, pin_memory=False)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_data, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_data, self.batch_size, shuffle=False, num_workers=self.num_workers)
