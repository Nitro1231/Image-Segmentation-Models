import sys
import torch
import torchmetrics
import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam
from argparse import Namespace

sys.path.append('.')
from models.unet import UNet
from models.double_unet import DoubleUNet
from models.r2_unet import R2UNet
from models.attention_unet import AttentionUNet
from models.attention_r2_unet import AttentionR2UNet


class SegmentationModel(pl.LightningModule):
    def __init__(self, args: Namespace) -> None:
        super(SegmentationModel, self).__init__()
        print('args:', args)
        self.save_hyperparameters()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = args.learning_rate

        match args.model_type:
            case 'UNet':
                self.model_class = UNet
            case 'DoubleUNet':
                self.model_class = DoubleUNet
            case 'R2UNet':
                self.model_class = R2UNet
            case 'AttentionUNet':
                self.model_class = AttentionUNet
            case 'AttentionR2UNet':
                self.model_class = AttentionR2UNet
            case _:
                raise ValueError('Model type is not recognizable.')

        self.model = self.model_class(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            n_encoders=args.n_encoders,
            embedding_size=args.embedding_size,
            kernel_size=args.kernel_size,
            stride=args.stride
        )

        num_classes = args.out_channels
        self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.train_iou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=num_classes)
        self.val_iou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=num_classes)
        self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes)
        self.val_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes)
    
    def forward(self, X):
        X = torch.nan_to_num(X)
        return self.model(X)
    
    def training_step(self, batch: tuple, batch_idx: int):
        image, mask = batch
        prediction = self.forward(image)
        loss = self.loss_fn(prediction, mask.float()) # Use original prediction for loss computation
        pred_class = torch.argmax(prediction, dim=1).float() # Use argmax only for metrics
        mask_class = torch.argmax(mask, dim=1).float() # Convert one-hot encoded mask to class indices
        self.log('train_accuracy', self.train_accuracy(pred_class, mask_class), on_epoch=True)
        self.log('train_iou', self.train_iou(pred_class, mask_class), on_epoch=True)
        self.log('train_f1', self.train_f1(pred_class, mask_class), on_epoch=True)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        image, mask = batch
        prediction = self.forward(image)
        loss = self.loss_fn(prediction, mask.float()) # Use original prediction for loss computation
        pred_class = torch.argmax(prediction, dim=1).float() # Use argmax only for metrics
        mask_class = torch.argmax(mask, dim=1).float() # Convert one-hot encoded mask to class indices
        self.log('val_accuracy', self.val_accuracy(pred_class, mask_class), on_epoch=True)
        self.log('val_iou', self.val_iou(pred_class, mask_class), on_epoch=True)
        self.log('val_f1', self.val_f1(pred_class, mask_class), on_epoch=True)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # Log detailed training metrics
        self.log('epoch_train_accuracy', self.train_accuracy.compute())
        self.log('epoch_train_iou', self.train_iou.compute())
        self.log('epoch_train_f1', self.train_f1.compute())
        print(f'Train Epoch {self.current_epoch}: Accuracy={self.train_accuracy.compute()}, IoU={self.train_iou.compute()}, F1={self.train_f1.compute()}')
        self.train_accuracy.reset()
        self.train_iou.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        # Log detailed validation metrics
        self.log('epoch_val_accuracy', self.val_accuracy.compute())
        self.log('epoch_val_iou', self.val_iou.compute())
        self.log('epoch_val_f1', self.val_f1.compute())
        print(f'Validation Epoch {self.current_epoch}: Accuracy={self.val_accuracy.compute()}, IoU={self.val_iou.compute()}, F1={self.val_f1.compute()}')
        self.val_accuracy.reset()
        self.val_iou.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
