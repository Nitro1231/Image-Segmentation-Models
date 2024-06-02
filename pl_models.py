import torch
import pytorch_lightning as pl
from models.unet import UNet
from models.double_unet import DoubleUNet
from models.r2_unet import R2UNet
from models.attention_unet import AttentionUNet
from models.attention_r2_unet import AttentionR2UNet
from torchmetrics.functional import accuracy, f1_score, jaccard_index


class BaseModel(pl.LightningModule):
    def __init__(self, model, criterion):
        super(BaseModel, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, masks = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, masks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, masks = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, masks)
        preds = outputs # (outputs > 0.5).int()
        acc = accuracy(preds, masks.int(), task='binary')
        iou = jaccard_index(preds, masks.int(), task='binary')
        f1 = f1_score(preds, masks.int(), task='binary')
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)


def create_model(model_name, n_encoders=2, embedding_size=64):
    if model_name == 'UNet':
        return UNet(in_channels=3, out_channels=1, n_encoders=n_encoders, embedding_size=embedding_size)
    elif model_name == 'R2UNet':
        return R2UNet(in_channels=3, out_channels=1, n_encoders=n_encoders, embedding_size=embedding_size)
    elif model_name == 'AttentionUNet':
        return AttentionUNet(in_channels=3, out_channels=1, n_encoders=n_encoders, embedding_size=embedding_size)
    elif model_name == 'AttentionR2UNet':
        return AttentionR2UNet(in_channels=3, out_channels=1, n_encoders=n_encoders, embedding_size=embedding_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
