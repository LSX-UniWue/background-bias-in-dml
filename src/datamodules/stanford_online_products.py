#! /usr/bin/env python3

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
import os
import logging
from tqdm import tqdm
from PIL import Image
import pytorch_lightning as pl
from typing import Union, Optional, List, Dict
from pytorch_metric_learning.samplers import MPerClassSampler
from pathlib import Path
import zipfile
from torchvision.transforms import ToTensor

class StanfordOnlineProducts(Dataset):
    url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
    filename = 'Stanford_Online_Products.zip'
    md5 = '7f73d41a2f44250d4779881525aea32e'

    filename_masked = 'SOP_masked.zip'

    def __init__(self, root, mode="train", transform=None, download=False):
        self.root = root
        self.mode = mode

        assert mode in ["train", "val", "test", "all"]

        if download:
            try:
                # self.set_paths_and_labels(assert_files_exist=True)
                self.set_paths_and_labels()
            except:
                self.download_dataset()
                self.set_paths_and_labels()
        else:
            self.set_paths_and_labels()
        
        self.transform = transform
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        # We need to paths to the mask paths.
        # data/Stanford_Online_Products/bicycle_final/111085122871_0.JPG
        # should be converted to
        # data/SOP_masked/111085122871_0.png
        path_mask = "/".join(path.split("/")[:-3] + ["SOP_masked"] + path.split("/")[-1:]).replace("JPG", "png")
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
        from pandas import read_csv

        # overall we have 11318 classes, we take approx. 3/4 of them for training
        train_val_split_label = 8489
        if self.mode in ["train", "val"]:
            info_files = {"train": "Ebay_train.txt"}
        elif self.mode == "test":
            info_files = {"test": "Ebay_test.txt"}
        else: # all
            info_files = {"train": "Ebay_train.txt", "test": "Ebay_test.txt"}
        
        self.img_paths = []
        self.labels = []
        self.super_labels = []
        global_idx = 0
        for k, v in info_files.items():
            curr_file = read_csv(os.path.join(self.dataset_folder, v), delim_whitespace=True, header=0).values
            self.img_paths.extend([os.path.join(self.dataset_folder, x) for x in list(curr_file[:,3])])
            self.labels.extend(list(curr_file[:,1] - 1))
            self.super_labels.extend(list(curr_file[:,2] - 1))
            global_idx += len(curr_file)
        self.labels = np.array(self.labels)
        self.super_labels = np.array(self.super_labels)

        if self.mode == "train":
            self.img_paths = [self.img_paths[i] for i in range(len(self.img_paths)) if self.labels[i] < train_val_split_label]
            self.labels = self.labels[self.labels < train_val_split_label]
        elif self.mode == "val":
            self.img_paths = [self.img_paths[i] for i in range(len(self.img_paths)) if self.labels[i] >= train_val_split_label]
            self.labels = self.labels[self.labels >= train_val_split_label]

        assert len(self.labels) == len(self.img_paths)

    def set_paths_and_labels(self, assert_files_exist=False):
        self.dataset_folder = os.path.join(self.root, "Stanford_Online_Products")
        self.load_labels()
        # assert len(np.unique(self.labels)) == 22634
        # assert self.__len__() == 120053

        # We just assume the files exist to avoid heavy load on the file system
        # if assert_files_exist:
        #     logging.info("Checking if dataset images exist")
        #     for x in tqdm.tqdm(self.img_paths):
        #         assert os.path.isfile(x)

    def download_dataset(self):
        download_url(self.url, self.root, filename=self.filename, md5=self.md5)
        with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zip_ref:
            zip_ref.extractall(self.root)
        with zipfile.ZipFile(os.path.join(self.root, self.filename_masked), 'r') as zip_ref:
            zip_ref.extractall(self.root)



class SOPDataModule(pl.LightningDataModule):
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
        self.num_train_classes = 8489
        self.m_per_class = m_per_class

    def prepare_data(self) -> None:
        # Download data if not present
        StanfordOnlineProducts(self.root, download=True)

    def setup(self, stage: Optional[str]) -> None:
        self.train_data = StanfordOnlineProducts(self.root, mode="train", transform=self.train_transform, download=False)
        self.val_data = StanfordOnlineProducts(self.root, mode="val", transform=self.val_transform, download=False)
        self.test_data = StanfordOnlineProducts(self.root, mode="test", transform=self.val_transform, download=False)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        # length_before_new_iter is set to 100 iterations per epoch, based on the `iterations_per_epoch` value from the powerful-benchmarker default config
        sampler = MPerClassSampler(self.train_data.labels, self.m_per_class, self.batch_size, length_before_new_iter=100*self.batch_size)
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler, drop_last=True, pin_memory=False)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_data, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_data, self.batch_size, shuffle=False, num_workers=self.num_workers)
