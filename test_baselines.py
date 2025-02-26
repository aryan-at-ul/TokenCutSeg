import os
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl

import sys
sys.path.append('/home/annatar/projects/test_vqa_bignn/src/baseline') #chnage this for your path

from src.baseline.model import Trans4PASS, UNET, DUNet, GraphUNet, SwinUnet, AttaNet
from src.utils.data_loading import create_dataloaders



def compute_iou(pred, target, smooth=1e-6):
    pred_inds = (pred == 1)
    target_inds = (target == 1)
    intersection = (pred_inds & target_inds).float().sum()
    union = (pred_inds | target_inds).float().sum()
    return (intersection + smooth) / (union + smooth)




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
    
    def test_step(self, batch, batch_idx):
        images, masks = batch  # Assumes batch is a tuple (images, masks)
        logits = self.forward(images)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)
        
        # Compute metrics.
        iou = compute_iou(preds, masks)
        dice = compute_dice(preds, masks)
        acc = compute_accuracy(preds, masks)
        
        # Log metrics.
        self.log('test/loss', loss, sync_dist=True)
        self.log('test/iou', iou, sync_dist=True)
        self.log('test/dice', dice, sync_dist=True)
        self.log('test/acc', acc, sync_dist=True)
        
        return {"test_loss": loss, "test_iou": iou, "test_dice": dice, "test_acc": acc}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Test a segmentation model checkpoint on a test set.")
    parser.add_argument('--test-data-dir', type=str, default = '/media/annatar/NewDrive/isic2018/test', help="Directory with test images")
    parser.add_argument('--test-mask-dir', type=str, default= '/media/annatar/NewDrive/isic2018/test_masks', help="Directory with test masks")
    parser.add_argument('--image-size', type=int, default=224, help="Image size to resize/crop to")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size for testing")
    parser.add_argument('--num-workers', type=int, default=2, help="Number of data loading workers")
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use (0 for CPU)")
    parser.add_argument('--precision', type=int, default=16, help="Precision (16 or 32)")
    parser.add_argument('--checkpoint-path', type=str,default='outputs_trans/checkpoints/isic_2018_baseline_trans-epoch=39-val/iou=0.87.ckpt', help="Path to the checkpoint file")
    parser.add_argument('--model', type=str, default='trans', choices=['unet', 'dunet', 'vig', 'attanet', 'swinunet', 'trans'],
                        help="Model architecture to use")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(42)
    
    # Select the model based on the argument.
    if args.model == 'unet':
        base_model = UNET(in_channels=3, out_channels=2, features=[64, 128, 256, 512])
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
    
    # Load the LightningModule from checkpoint.
    # The checkpoint should have been saved using the same LightningModule.
    model = SegmentationLightningModule.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model=base_model,
        learning_rate=1e-4  # (ensure this matches your training config)
    )
    
    # Create test dataloader.
    # Here we use the same data-loading function; by passing test directories as both training and validation,
    # we simply extract the validation loader.
    _, test_loader = create_dataloaders(
        train_data_dir=args.test_data_dir,
        train_mask_dir=args.test_mask_dir,
        val_data_dir=args.test_data_dir,
        val_mask_dir=args.test_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    # Initialize the trainer.
    trainer = pl.Trainer(
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus,
        precision=args.precision,
        logger=False  # disable logging if you prefer
    )
    
    # Run testing.
    results = trainer.test(model, test_loader)
    
    print("Test results:")
    print(results)


if __name__ == "__main__":
    main()
