import os
import torch
import argparse
import pytorch_lightning as pl
from tqdm import tqdm
from torchmetrics import JaccardIndex, Dice
import torch.nn.functional as F
from src.vq_seg_model import VQGPTSegmentation
from src.utils.data_loading import create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--test-data-dir', type=str, default='/media/annatar/NewDrive/isic2018/test',
                        help='Directory containing test images')
    parser.add_argument('--test-mask-dir', type=str, default='/media/annatar/NewDrive/isic2018/test_masks',
                        help='Directory containing test masks')
    parser.add_argument('--image-size', type=int, default=224)
    
    # Model arguments
    parser.add_argument('--checkpoint-path', type=str, 
                        default='segmentation_training_2018/checkpoints/segmentation_isic2018-epoch=20-val_iou=0.0000.ckpt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Model config
    parser.add_argument('--vq-embed-dim', type=int, default=256)
    parser.add_argument('--vq-n-embed', type=int, default=1024)
    parser.add_argument('--vq-hidden-channels', type=int, default=128)
    parser.add_argument('--vq-n-res-blocks', type=int, default=2)
    parser.add_argument('--gpt-n-layer', type=int, default=12)
    parser.add_argument('--gpt-n-head', type=int, default=4)
    parser.add_argument('--gpt-n-embd', type=int, default=256)
    
    return parser.parse_args()

@torch.no_grad()
def generate_prediction(model, images):
    """Generate segmentation prediction using the model's components."""
    images_norm = 2.0 * images - 1.0
    
    quant, _, info = model.vqgan.encode(images_norm)
    if isinstance(info, tuple) and len(info) >= 3:
        indices = info[2]
    else:
        indices = info
    
    B, C, H, W = quant.shape
    indices = indices.view(B, H * W)
    
    token_features = model.gpt(indices, return_features=True)
    S_all, _ = model.deepcut_module(token_features)
    S_all = S_all.transpose(1, 2).view(B, model.hparams.segmentation_config['num_classes'], H, W)
    
    seg_logits = F.interpolate(S_all, size=images.shape[2:], mode='bilinear', align_corners=False)
    predictions = torch.argmax(seg_logits, dim=1)
    
    return predictions

def main():
    args = parse_args()
    
    vqgan_config = {
        'n_embed': args.vq_n_embed,
        'embed_dim': args.vq_embed_dim,
        'hidden_channels': args.vq_hidden_channels,
        'n_res_blocks': args.vq_n_res_blocks,
    }
    
    gpt_config = {
        'vocab_size': args.vq_n_embed + 1,
        'block_size': (args.image_size // 4) ** 2,
        'n_layer': args.gpt_n_layer,
        'n_head': args.gpt_n_head,
        'n_embd': args.gpt_n_embd
    }
    
    segmentation_config = {
        'num_classes': 2,  # Binary segmentation
        'use_deepcut': True,
        'deepcut_loss_weight': 0.2,
    }
    
    jaccard = JaccardIndex(task="multiclass", num_classes=2).cuda()
    dice_metric = Dice(num_classes=2, average='macro').cuda()
    
    # Custom metric functions
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
    
    # Load model
    model = VQGPTSegmentation.load_from_checkpoint(
        args.checkpoint_path,
        vqgan_path='vqgan_model_pretrained.ckpt',
        vqgan_config=vqgan_config,
        gpt_config=gpt_config,
        segmentation_config=segmentation_config,
        strict=False
    )
    model.eval()
    model.cuda()
    
    # Create test dataloader
    _, test_loader = create_dataloaders(
        train_data_dir=args.test_data_dir,
        train_mask_dir=args.test_mask_dir,
        val_data_dir=args.test_data_dir,
        val_mask_dir=args.test_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    # Test loop
    custom_iou_scores = []
    custom_dice_scores = []
    custom_acc_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images, masks = batch
            images = images.cuda()
            masks = masks.cuda()
            
            try:
                predictions = generate_prediction(model, images)
                
                jaccard.update(predictions, masks)
                dice_metric.update(predictions, masks)
                
                custom_iou = compute_iou(predictions, masks)
                custom_dice = compute_dice(predictions, masks)
                custom_acc = compute_accuracy(predictions, masks)
                
                custom_iou_scores.append(custom_iou.item())
                custom_dice_scores.append(custom_dice.item())
                custom_acc_scores.append(custom_acc.item())
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    

    final_iou = jaccard.compute()
    final_dice = dice_metric.compute()
    
    
    print("\nResults for ISIC Test Set:")
    print(f"IoU Score: {final_iou:.4f}")
    print(f"Dice Score: {final_dice:.4f}")
    

if __name__ == "__main__":
    main()