import sys
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary

sys.path.append('.')
from src.visual import *
from src.model import SegmentationModel
from src.datamodule import BrainScanDataModule


def generate_images(args: Namespace, model, data_module, indices):
    np.random.seed(1)
    h5_files = data_module.h5_files

    for i in indices:
        image, mask = data_module.get_image_mask(h5_files[i])
        model.eval()
        with torch.no_grad():
            image_tensor = torch.tensor(image).unsqueeze(0).float()
            predicted_mask = model(image_tensor).squeeze().numpy()

        display_image_channels(f'{args.output_dir}/image_{i}.png', image)
        display_mask_channels_as_rgb(f'{args.output_dir}/mask_{i}.png', mask)
        display_mask_channels_as_rgb(f'{args.output_dir}/predicted_mask_{i}.png', predicted_mask)
        overlay_masks_on_image(f'{args.output_dir}/mask_on_image_{i}.png', image, mask)
        overlay_masks_on_image(f'{args.output_dir}/predicted_mask_on_image_{i}.png', image, predicted_mask)

def evaluate(args: Namespace) -> None:
    print('args:', args)

    # Set the seed for reproducibility
    # seed_everything(42, workers=True)

    model = SegmentationModel.load_from_checkpoint(args.checkpoint_path, args=args)

    data_module = BrainScanDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    callbacks = [
        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]

    trainer = pl.Trainer(
        accelerator='auto',
        callbacks=callbacks,
        logger=False
    )

    trainer.test(model, datamodule=data_module)
    generate_images(args, model, data_module, indices=[1178, 1200, 1356])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('-m', '--model_type', type=str, default='UNet', help='The model to initialize.')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='Number of workers for dataloader.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='The learning rate for training model.')
    parser.add_argument('-e', '--max_epochs', type=int, default=12, help='Number of epochs to train for.')
    parser.add_argument('--in_channels', type=int, default=4, help='Number of input channels.')
    parser.add_argument('--out_channels', type=int, default=3, help='Number of output channels.')
    parser.add_argument('--n_encoders', type=int, default=8, help='Number of encoders.')
    parser.add_argument('--embedding_size', type=int, default=32, help='Embedding size of the neural network.')
    parser.add_argument('--kernel_size', type=int, default=2, help='Kernel size of the neural network.')
    parser.add_argument('--stride', type=int, default=2, help='Stride size of the neural network.')
    parser.add_argument('--data_dir', type=str, default='./data/BraTS2020_training_data/content/data', help='Dataset folder path.')

    evaluate(parser.parse_args())
