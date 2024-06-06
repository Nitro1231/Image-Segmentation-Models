import os
import sys
import random
import numpy as np

sys.path.append('.')
from src.visual import *
from src.datamodule import BrainScanDataModule

np.random.seed(1)

if __name__ == '__main__':
    data_dir = './data/BraTS2020_training_data/content/data'
    data_module = BrainScanDataModule(data_dir=data_dir)
    data_module.prepare_data()
    # data_module.print_info()

    h5_files = data_module.h5_files
    for i in [1178, 1200, 1356]:
        sample_file_path = os.path.join(data_dir, h5_files[i])
        image, mask = get_image_mask(sample_file_path)

        # View images using plotting functions
        display_image_channels(f'./plot/image_{i}.png', image)
        display_mask_channels_as_rgb(f'./plot/mask_{i}.png', mask)
        overlay_masks_on_image(f'./plot/mask_on_image_{i}.png', image, mask)
