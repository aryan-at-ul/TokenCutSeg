import os
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

import sys
sys.path.append('/home/annatar/projects/test_vqa_bignn/src/baseline') #chnage this for your path


from src.baseline.model import Trans4PASS, UNET, DUNet, GraphUNet, SwinUnet, AttaNet
from src.utils.data_loading import create_dataloaders


def compute_iou(pred, target, num_classes=2, smooth=1e-6):
    """Compute the mean Intersection-over-Union over all classes."""
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()
        ious.append((intersection + smooth) / (union + smooth))
    return sum(ious) / len(ious)


def compute_dice(pred, target, num_classes=2, smooth=1e-6):
    """Compute the mean Dice score over all classes."""
    dice_scores = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).float().sum()
        dice_score = (2 * intersection + smooth) / (pred_inds.float().sum() + target_inds.float().sum() + smooth)
        dice_scores.append(dice_score)
    return sum(dice_scores) / len(dice_scores)


def compute_accuracy(pred, target):
    """Compute pixelwise accuracy."""
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    return correct / total


class SegmentationLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        """
        Wraps a segmentation model.

        Args:
            model: A segmentation model that returns raw logits of shape (B, num_classes, H, W).
            learning_rate: The optimizer learning rate.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch  # Assumes batch is a tuple (images, masks)
        logits = self.forward(images)
        loss = self.criterion(logits, masks)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self.forward(images)
        loss = self.criterion(logits, masks)
        # Get predicted segmentation by taking argmax along the channel dim.
        preds = torch.argmax(logits, dim=1)
        
        # Compute metrics.
        iou = compute_iou(preds, masks)
        acc = compute_accuracy(preds, masks)
        dice = compute_dice(preds, masks)
        
        # Log the metrics.
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/iou', iou, prog_bar=True, sync_dist=True)
        self.log('val/acc', acc, prog_bar=True, sync_dist=True)
        self.log('val/dice', dice, prog_bar=True, sync_dist=True)
        
        # Log sample images only from the first batch.
        if batch_idx == 0:
            self._log_validation_images(images, preds, masks)
        
        return {"val_loss": loss, "iou": iou, "acc": acc, "dice": dice}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def _log_validation_images(self, images, pred_masks, masks):
        """
        Logs a few validation examples to WandB.
        Assumes that images are in the range [-1, 1] and converts them to [0, 1].
        """
        # Normalize images from [-1, 1] to [0, 1]
        images_norm = (images + 1.0) / 2.0

        self.logger.experiment.log({
            "val/examples": [
                wandb.Image(img.cpu(), caption=f"Sample {i}") 
                for i, img in enumerate(images_norm[:4])
            ],
            "val/predictions": [
                wandb.Image(pred.unsqueeze(0).float().cpu(), caption=f"Prediction {i}")
                for i, pred in enumerate(pred_masks[:4])
            ],
            "val/masks": [
                wandb.Image(mask.unsqueeze(0).float().cpu(), caption=f"Mask {i}")
                for i, mask in enumerate(masks[:4])
            ],
            "global_step": self.global_step
        })


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data arguments , isic2017 > /media/annatar/NewDrive/ISIC 2017 for segmentation/, 2026> /home/annatar/projects/datasets/ISIC2016/
    # isic2018> /media/annatar/NewDrive/isic2018/
    parser.add_argument('--train-data-dir', type=str, default='/media/annatar/NewDrive/isic2018/train')
    parser.add_argument('--train-mask-dir', type=str, default='/media/annatar/NewDrive/isic2018/train_masks')
    parser.add_argument('--val-data-dir', type=str, default='/media/annatar/NewDrive/isic2018/val')
    parser.add_argument('--val-mask-dir', type=str, default='/media/annatar/NewDrive/isic2018/val_masks')
    parser.add_argument('--image-size', type=int, default=224)
    

    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    

    # parser.add_argument('--vq-embed-dim', type=int, default=256)
    # parser.add_argument('--vq-n-embed', type=int, default=1024)
    # parser.add_argument('--vq-hidden-channels', type=int, default=128)
    # parser.add_argument('--vq-n-res-blocks', type=int, default=2)
    # parser.add_argument('--disc-start', type=int, default=10000)
    # parser.add_argument('--disc-weight', type=float, default=0.8)
    # parser.add_argument('--perceptual-weight', type=float, default=1.0)
    # parser.add_argument('--codebook-weight', type=float, default=1.0)
    
    # parser.add_argument('--gpt-n-layer', type=int, default=12)
    # parser.add_argument('--gpt-n-head', type=int, default=4)
    # parser.add_argument('--gpt-n-embd', type=int, default=256)
    
    parser.add_argument('--output-dir', type=str, default='outputs_trans')
    parser.add_argument('--experiment-name', type=str, default='isic_2018_baseline_trans')
    

    parser.add_argument('--model', type=str, default='trans', choices=['unet', 'dunet', 'vig', 'attanet', 'swinunet', 'trans'])
    
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(42)
    
    os.makedirs(args.output_dir, exist_ok=True)
    

    if args.model == 'unet':
        base_model = UNET(in_channels=3, out_channels=2, features=[64, 128, 256, 512, 1024])
    elif args.model == 'dunet':
        base_model = DUNet(num_classes=2)
    elif args.model == 'vig':
        base_model = GraphUNet(3, 2)
    elif args.model == 'attanet':
        base_model = AttaNet(n_classes=2)
    elif args.model == 'swinunet':
        base_model = SwinUnet()
    elif args.model == 'trans':
        base_model = Trans4PASS()
    else:
        raise ValueError("Invalid model selection")
    
    model = SegmentationLightningModule(model=base_model, learning_rate=args.learning_rate)
    

    train_loader, val_loader = create_dataloaders(
        train_data_dir=args.train_data_dir,
        train_mask_dir=args.train_mask_dir,
        val_data_dir=args.val_data_dir,
        val_mask_dir=args.val_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename=f'{args.experiment_name}' + '-{epoch:02d}-{val/iou:.2f}',
        monitor='val/iou',
        mode='max',
        save_top_k=1,
        save_last=False,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    

    wandb_logger = WandbLogger(
        project=args.experiment_name,
        name=f"{args.experiment_name}-{args.image_size}-bs{args.batch_size}-lr{args.learning_rate}"
    )
    

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10,
        strategy='auto'
    )
    
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
