import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class BrainScanDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
        np.random.shuffle(self.data_paths)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        with h5py.File(file_path, 'r') as file:
            image = file['image'][()]
            mask = file['mask'][()]

            # Reshape: (H, W, C) -> (C, H, W)
            image = image.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))

            # Adjusting pixel values for each channel in the image so they are between 0 and 255
            for i in range(image.shape[0]):             # Iterate over channels
                min_val = np.min(image[i])              # Find the min value in the channel
                image[i] = image[i] - min_val           # Shift values to ensure min is 0
                max_val = np.max(image[i]) + 1e-4       # Find max value to scale max to 1 now.
                image[i] = image[i] / max_val

            # Convert to float and scale the whole image
            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32) 

        return image, mask
