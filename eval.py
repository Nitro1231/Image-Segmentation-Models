import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pl_models import BaseModel, create_model

# Dataset loading and transformations

def train():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Download and prepare the dataset
    dataset = datasets.OxfordIIITPet(
        root='./data',
        split='trainval',
        target_types='segmentation',
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #, num_workers=9, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) #, num_workers=9, persistent_workers=True)

    # Models to test
    model_names = ['UNet'] #, 'R2UNet', 'AttentionUNet', 'AttentionR2UNet']
    results = {}
    sample_images = {}

    for model_name in model_names:
        model = create_model(model_name)
        criterion = nn.CrossEntropyLoss()
        lightning_model = BaseModel(model, criterion)

        # Training
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator='auto',
            callbacks=[
                ModelCheckpoint(monitor='val_loss'),
                EarlyStopping(monitor='val_loss', patience=3)
            ]
        )
        trainer.fit(lightning_model, train_loader, val_loader)

        # Evaluation
        metrics = trainer.validate(lightning_model, val_loader)
        results[model_name] = metrics[0]

        val_batch = next(iter(val_loader))
        inputs, masks = val_batch

        print('input shape:', inputs.shape)
        print('mask shape:', masks.shape)

        outputs = lightning_model(inputs)

        sample_images[model_name] = {'input': inputs[0], 'mask': masks[0], 'prediction': outputs[0]}


    # Plotting the results
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    model_names = list(results.keys())
    accuracies = [results[model]['val_acc'] for model in model_names]
    ious = [results[model]['val_iou'] for model in model_names]
    f1s = [results[model]['val_f1'] for model in model_names]
    # training_times = [trainer.logged_metrics['train_loss'].to(torch.device('cpu')).numpy() for model in model_names]

    ax[0, 0].bar(model_names, accuracies)
    ax[0, 0].set_title('Accuracy')

    ax[0, 1].bar(model_names, ious)
    ax[0, 1].set_title('IOU')

    ax[1, 0].bar(model_names, f1s)
    ax[1, 0].set_title('F1 Score')

    # ax[1, 1].bar(model_names, training_times)
    # ax[1, 1].set_title('Training Time')

    plt.tight_layout()
    plt.show()

    # Plotting sample images and predictions
    fig, axes = plt.subplots(len(model_names)+1, 3, figsize=(15, 5 * len(model_names)))
    for i, model_name in enumerate(model_names):
        input_img = sample_images[model_name]['input'].cpu().numpy().transpose(1, 2, 0)
        mask_img = sample_images[model_name]['mask'].cpu().numpy().squeeze()
        pred_img = sample_images[model_name]['prediction'].detach().numpy().squeeze()

        axes[i, 0].imshow(input_img)
        axes[i, 0].set_title(f'{model_name} - Input Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_img, cmap='gray')
        axes[i, 1].set_title(f'{model_name} - Ground Truth Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_img, cmap='gray')
        axes[i, 2].set_title(f'{model_name} - Predicted Mask')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train()