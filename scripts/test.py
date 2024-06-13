import os
import sys
import random
import numpy as np

sys.path.append('.')
from src.visual import *
from src.datamodule import BrainScanDataModule


PATH = './plot/test'


if __name__ == '__main__':
    np.random.seed(1)
    data_dir = './data/BraTS2020_training_data/content/data'
    data_module = BrainScanDataModule(data_dir=data_dir)
    data_module.prepare_data()
    data_module.print_info()

    h5_files = data_module.h5_files
    for i in [1178, 1200, 1356]:
        image, mask = get_image_mask(h5_files[i])

        # View images using plotting functions
        display_image_channels(f'{PATH}/image_{i}.png', image)
        display_mask_channels_as_rgb(f'{PATH}/mask_{i}.png', mask)
        overlay_masks_on_image(f'{PATH}/mask_on_image_{i}.png', image, mask)
