import h5py
import numpy as np
import matplotlib.pyplot as plt
from .constants import IMAGE_NAMES, MASK_NAMES
plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = '#171717'
plt.rcParams['text.color']       = '#DDDDDD'


def get_image_mask(path: str) -> tuple[np.ndarray]:
    with h5py.File(path, 'r') as file:
        image = file['image'][()].transpose(2, 0, 1)
        mask = file['mask'][()].transpose(2, 0, 1)
        return image, mask

def display_image_channels(path: str, image: np.ndarray, title: str = 'Image Channels') -> None:
    channel_names = IMAGE_NAMES
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, (ax, name) in enumerate(zip(axes.flatten(), channel_names)):
        channel_image = image[idx, :, :] # Transpose the array to display the channel
        ax.imshow(channel_image, cmap='magma')
        ax.axis('off')
        ax.set_title(name)
    plt.tight_layout()
    plt.suptitle(title, fontsize=20)
    plt.savefig(path, dpi=300)

def display_mask_channels_as_rgb(path: str, mask: np.ndarray, title: str = 'Mask Channels as RGB') -> None:
    channel_names = MASK_NAMES
    fig, axes = plt.subplots(1, 3, figsize=(9.75, 5))
    for idx, (ax, name) in enumerate(zip(axes, channel_names)):
        rgb_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        rgb_mask[..., idx] = mask[idx, :, :] * 255 # Transpose the array to display the channel
        ax.imshow(rgb_mask)
        ax.axis('off')
        ax.set_title(name)
    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(path, dpi=300)

def overlay_masks_on_image(path: str, image: np.ndarray, mask: np.ndarray, title: str = 'Brain MRI with Tumour Masks Overlay') -> None:
    t1_image = image[0, :, :] # Use the first channel of the image
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())

    rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)
    color_mask = np.stack([mask[0, :, :], mask[1, :, :], mask[2, :, :]], axis=-1)
    rgb_image = np.where(color_mask, color_mask, rgb_image)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.title(title, fontsize=18)
    plt.axis('off')
    plt.savefig(path, dpi=300)
