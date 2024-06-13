import sys
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary
)

sys.path.append('.')
from src.model import SegmentationModel
from src.datamodule import BrainScanDataModule


def train(args: Namespace) -> None:
    print('args:', args)

    model = SegmentationModel(args)

    data_module = BrainScanDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=f'checkpoints/{args.model_type}',
            filename='{epoch}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]

    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        # logger=logger
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
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
    
    train(parser.parse_args())
