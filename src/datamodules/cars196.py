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


class Cars196(Dataset):
    ims_url = 'http://imagenet.stanford.edu/internal/car196/car_ims.tgz'
    ims_filename = 'car_ims.tgz'
    ims_md5 = 'd5c8f0aa497503f355e17dc7886c3f14'

    masked_filename = 'car_ims_masked.tgz'

    annos_url = 'http://imagenet.stanford.edu/internal/car196/cars_annos.mat'
    annos_filename = 'cars_annos.mat'
    annos_md5 = 'b407c6086d669747186bd1d764ff9dbc'

    def __init__(self, root="./data", mode="train", transform=None, download=False):
        self.root = os.path.join(root, "cars196")
        self.mode = mode
        self.to_tensor = ToTensor()

        assert self.mode in ["train", "val", "test", "all"]

        if download:
            try:
                self.set_paths_and_labels(assert_files_exist=True)
            except:
                self.download_dataset()
                self.set_paths_and_labels()
        else:
            self.set_paths_and_labels()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        path_mask = path.replace("car_ims", "car_ims_masked").replace("jpg", "png")
        mask = Image.open(path_mask).convert("L")
        label = self.labels[idx]
        if self.transform is not None:
            img, mask = self.transform(img, mask)
            
            assert img.shape == (3, 227, 227), f"Image not in correct shape: {img.shape}"
            assert mask.shape == (1, 227, 227), f"Mask not in correct shape: {mask.shape}"

        return {
            "inputs": img,
            "labels": label,
            "masks": mask
        }

    def load_labels(self):
        img_data = sio.loadmat(os.path.join(self.dataset_folder, "cars_annos.mat"))
        # labels start at 1, so we subtract 1
        self.labels = np.array([int(i[0, 0]) for i in img_data["annotations"]["class"][0]]) - 1
        self.img_paths = [os.path.join(self.dataset_folder, i[0]) for i in img_data["annotations"]["relative_im_path"][0]]

        if self.mode == "train":
            r = list(range(74))
        elif self.mode == "val":
            r = list(range(74, 98))
        elif self.mode == "test":
            r = list(range(98, 196))
        else: # all
            r = list(set(self.labels))

        self.img_paths = [self.img_paths[i] for i in range(len(self.img_paths)) if self.labels[i] in r]
        self.labels = [l for l in self.labels if l in r]

        self.class_names = [i[0] for i in img_data["class_names"][0]]

    def set_paths_and_labels(self, assert_files_exist=False):
        self.dataset_folder = self.root
        self.load_labels()
        if assert_files_exist:
            logging.info("Checking if dataset images exist")
            for x in tqdm(self.img_paths):
                assert os.path.isfile(x)

    def download_dataset(self):
        url_infos = [(self.ims_url, self.ims_filename, self.ims_md5), 
                    (self.annos_url, self.annos_filename, self.annos_md5)]
        for url, filename, md5 in url_infos:
            download_url(url, self.root, filename=filename, md5=md5)
        with tarfile.open(os.path.join(self.root, self.ims_filename), "r:gz") as tar:
            tar.extractall(path=self.root)
        with tarfile.open(os.path.join(self.root, self.masked_filename), "r:gz") as tar:
            tar.extractall(path=self.root)




class Cars196DataModule(pl.LightningDataModule):
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
        self.num_train_classes = 74
        self.m_per_class = m_per_class

    def prepare_data(self) -> None:
        # Download data if not present
        Cars196(self.root, download=True)

    def setup(self, stage: Optional[str]) -> None:
        # if stage in ["fit", None]:
        self.train_data = Cars196(self.root, mode="train", transform=self.train_transform, download=False)
        self.val_data = Cars196(self.root, mode="val", transform=self.val_transform, download=False)

        # if stage in ["test", None]:
        self.test_data = Cars196(self.root, mode="test", transform=self.val_transform, download=False)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        # length_before_new_iter is set to 100 iterations per epoch, based on the `iterations_per_epoch` value from the powerful-benchmarker default config
        sampler = MPerClassSampler(self.train_data.labels, self.m_per_class, batch_size=self.batch_size, length_before_new_iter=100*self.batch_size)
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler, drop_last=True, pin_memory=False)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_data, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_data, self.batch_size, shuffle=False, num_workers=self.num_workers)
