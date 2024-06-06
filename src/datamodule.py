import os
import h5py
import numpy as np
import pytorch_lightning as pl
from .dataset import BrainScanDataset
from torch.utils.data import DataLoader


class BrainScanDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, val_split=0.2, num_workers=8):
        super(BrainScanDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.h5_files = list()

    def prepare_data(self):
        # This is called only once and on only one GPU
        h5_files = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]
        np.random.shuffle(h5_files)
        self.h5_files = h5_files

    def setup(self, stage=None):
        # This is called on every GPU
        split_idx = int((1 - self.val_split) * len(self.h5_files))
        train_files = self.h5_files[:split_idx]
        val_files = self.h5_files[split_idx:]

        self.train_dataset = BrainScanDataset(train_files)
        self.val_dataset = BrainScanDataset(val_files, deterministic=True)

    def print_info(self):
        print('Number of `.h5` files:', len(self.h5_files))
        print('Example file names:', self.h5_files[:3])
        print()

        file_path = os.path.join(self.data_dir, self.h5_files[0])
        with h5py.File(file_path, 'r') as file:
            print('Keys for each file:', list(file.keys()))
            for key in file.keys():
                print(f'Data type of {key}:', type(file[key][()]))
                print(f'Shape of {key}:', file[key].shape)
                print('Array dtype:', file[key].dtype)
                print('Array max val:', np.max(file[key]))
                print('Array min val:', np.min(file[key]))
                print()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
