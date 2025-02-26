import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex
import wandb

class ConvSegHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        return self.conv_layers(x)

class VQSegmentationModel(pl.LightningModule):
    def __init__(self, vqgan_path: str, vqgan_config: dict, gpt_config: dict, 
                 segmentation_config: dict, learning_rate: float = 1e-4,
                 use_gpt: bool = True, use_mlm: bool = True):
        super().__init__()
        self.save_hyperparameters()
        
        # Store configuration flags
        self.use_gpt = use_gpt
        self.use_mlm = use_mlm and use_gpt  # MLM requires GPT
        
        # Load and freeze pretrained VQGAN
        from src.vqa.vqagan import VQGAN
        self.vqgan = VQGAN(**vqgan_config)
        checkpoint = torch.load(vqgan_path, map_location=self.device)
        state_dict = {k.replace('vqgan.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.vqgan.load_state_dict(state_dict)
        for param in self.vqgan.parameters():
            param.requires_grad = False
        self.vqgan.eval()

        # Initialize GPT and MLM components conditionally
        if self.use_gpt:
            from src.imagegpt_model import ImageGPT
            self.gpt = ImageGPT(**gpt_config)
            self.feature_dim = gpt_config['n_embd']
            
            if self.use_mlm:
                self.lm_head = nn.Linear(gpt_config['n_embd'], gpt_config['vocab_size'])
                self.mask_prob = gpt_config.get('mask_prob', 0.15)
                self.mask_token_id = gpt_config.get('mask_token_id', 0)
                self.gpt_loss_weight = gpt_config.get('loss_weight', 1.0)
        else:
            # Direct VQGAN features to segmentation head
            self.feature_dim = vqgan_config.get('embed_dim', 256)
        
        # Initialize segmentation head
        self.seg_head = ConvSegHead(
            in_channels=self.feature_dim,
            num_classes=segmentation_config['num_classes'],
            hidden_dim=128
        )

        # Metrics
        self.val_accuracy = Accuracy(task="multiclass", num_classes=segmentation_config['num_classes'])
        self.val_iou = JaccardIndex(task="multiclass", num_classes=segmentation_config['num_classes'])
        self.learning_rate = learning_rate
        
        # Print model configuration
        print(f"Model configuration: VQGAN > {'GPT > ' if use_gpt else ''}{'MLM > ' if use_mlm else ''}Segmentation")
        print(f"Feature dimension: {self.feature_dim}, Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def compute_dice(self, preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute Dice coefficient."""
        num_classes = self.hparams.segmentation_config['num_classes']
        dice_score = 0.0
        for c in range(num_classes):
            pred_c = (preds == c).float()
            target_c = (targets == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_score += (2 * intersection + eps) / (union + eps)
        return dice_score / num_classes

    def get_vqgan_output(self, images_norm):
        """Helper function to process VQGAN output consistently."""
        with torch.no_grad():
            quant, _, info = self.vqgan.encode(images_norm)
            if isinstance(info, tuple) and len(info) >= 3:
                indices = info[2]
            else:
                indices = info
            B, C, H, W = quant.shape
            return quant, indices.view(B, H * W), B, H, W

    def training_step(self, batch, batch_idx):
        images, masks = batch
        images = images.to(self.device)
        images_norm = 2.0 * images - 1.0

        # Extract VQGAN outputs
        quant, indices, B, H, W = self.get_vqgan_output(images_norm)
        
        # Path 1: VQGAN > Segmentation
        if not self.use_gpt:
            # Use quantized features directly
            features = quant  # Already [B, C, H, W] format
            lm_loss = torch.tensor(0.0, device=self.device)
        
        # Path 2: VQGAN > GPT[+MLM] > Segmentation
        else:
            if self.use_mlm and self.training:
                # Apply masking for MLM training
                rand = torch.rand(indices.shape, device=indices.device)
                mask = rand < self.mask_prob
                masked_indices = indices.clone()
                masked_indices[mask] = self.mask_token_id
                
                # Get features and token predictions
                token_features = self.gpt(masked_indices, return_features=True)
                lm_logits = self.lm_head(token_features)
                lm_loss = F.cross_entropy(lm_logits[mask], indices[mask]) if mask.any() else torch.tensor(0.0, device=self.device)
            else:
                # No masking, just get features
                token_features = self.gpt(indices, return_features=True)
                lm_loss = torch.tensor(0.0, device=self.device)
            
            # Reshape token features for 2D convolution: [B, H*W, C] -> [B, C, H, W]
            features = token_features.transpose(1, 2).reshape(B, -1, H, W)
        
        # Segmentation head (common for all paths)
        seg_logits = self.seg_head(features)
        
        # Upsample predictions to match target size
        seg_logits = F.interpolate(seg_logits, size=images.shape[2:], mode='bilinear', align_corners=False)
        
        # Compute segmentation loss
        seg_loss = F.cross_entropy(seg_logits, masks)
        
        # Combined loss based on model configuration
        if self.use_mlm:
            total_loss = self.gpt_loss_weight * lm_loss + seg_loss
        else:
            total_loss = seg_loss

        # Calculate metrics
        with torch.no_grad():
            preds = torch.argmax(seg_logits, dim=1)
            train_dice = self.compute_dice(preds, masks)

        # Logging
        log_dict = {
            'train/seg_loss': seg_loss,
            'train/total_loss': total_loss,
            'train/dice': train_dice,
        }
        
        if self.use_mlm:
            log_dict['train/lm_loss'] = lm_loss
            
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        images = images.to(self.device)
        images_norm = 2.0 * images - 1.0

        with torch.no_grad():
            # Get VQGAN outputs
            quant, indices, B, H, W = self.get_vqgan_output(images_norm)
            
            # Path 1: VQGAN > Segmentation
            if not self.use_gpt:
                # Use quantized features directly
                features = quant  # Already [B, C, H, W] format
            
            # Path 2: VQGAN > GPT > Segmentation
            else:
                # Get features without masking
                token_features = self.gpt(indices, return_features=True)
                # Reshape token features for 2D convolution: [B, H*W, C] -> [B, C, H, W]
                features = token_features.transpose(1, 2).reshape(B, -1, H, W)
            
            # Segmentation head (common for all paths)
            seg_logits = self.seg_head(features)
            
            # Upsample predictions
            seg_logits = F.interpolate(seg_logits, size=images.shape[2:], mode='bilinear', align_corners=False)
            preds = torch.argmax(seg_logits, dim=1)
            
            # Compute metrics
            val_dice = self.compute_dice(preds, masks)
            self.val_accuracy(preds, masks)
            self.val_iou(preds, masks)
            
            # Log metrics
            self.log_dict({
                'val/dice': val_dice,
                'val/accuracy': self.val_accuracy,
                'val/iou': self.val_iou,
            }, prog_bar=True, on_step=False, on_epoch=True)

            # Log visualizations periodically
            if batch_idx == 0 or batch_idx % 100 == 0:
                self._log_validation_images(images, preds, masks)

    def _log_validation_images(self, images, pred_masks, masks):
        """Log validation images to wandb."""
        images = (images + 1.0) / 2.0  # Denormalize
        num_vis = min(4, images.shape[0])
        
        vis_images = []
        for i in range(num_vis):
            vis_images.extend([
                wandb.Image(images[i].cpu(), caption=f"Sample {i}"),
                wandb.Image(pred_masks[i].float().cpu(), caption=f"Prediction {i}"),
                wandb.Image(masks[i].float().cpu(), caption=f"Ground Truth {i}")
            ])
        
        self.logger.experiment.log({
            "val/visualizations": vis_images,
            "global_step": self.global_step
        })

    def configure_optimizers(self):
        """Configure optimizers with cosine learning rate schedule."""
        params = []
        
        # Add parameters to optimize based on configuration
        if self.use_gpt:
            params.append({'params': self.gpt.parameters()})
            if self.use_mlm:
                params.append({'params': self.lm_head.parameters()})
                
        params.append({'params': self.seg_head.parameters()})
        
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }