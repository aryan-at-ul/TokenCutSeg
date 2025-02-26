import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex
from contextlib import contextmanager
import wandb
import math
import numpy as np
from src.imagegpt_model import ImageGPT
# PyG imports
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph, GCNConv
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
from src.vqa.vqagan import VQGAN


##############################################
# Helper: Scatter Mean (custom implementation)
##############################################
def scatter_mean(src, index, dim=0, dim_size=None):
    """
    Custom implementation of scatter mean aggregation.
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    out = torch.zeros((dim_size,) + src.shape[1:], dtype=src.dtype, device=src.device)
    ones = torch.ones_like(src[:, 0])
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index, ones)
    out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)
    count = count.clamp(min=1).unsqueeze(-1)
    out = out / count
    return out

##############################################
# RELATIVE CONV LAYER WITH BIDIRECTIONAL PASSING
##############################################
class RelConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RelConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.root = nn.Linear(in_channels, out_channels)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.xavier_uniform_(self.root.weight)
        if hasattr(self.root, 'bias') and self.root.bias is not None:
            nn.init.zeros_(self.root.bias)
        
    def forward(self, x, edge_index):
        """
        Improved bidirectional message passing.
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: [2, num_edges]
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        source, target = edge_index
        # Propagate from source -> target
        x_source_to_target = self.propagate_s2t(x, source, target)
        # Propagate from target -> source (reverse)
        x_target_to_source = self.propagate_s2t(x, target, source)
        x_out = self.root(x) + x_source_to_target + x_target_to_source
        return x_out
    
    def propagate_s2t(self, x, source, target):
        x_transformed = self.lin1(x)
        messages = x_transformed[source]
        aggr_out = scatter_mean(messages, target, dim=0, dim_size=x.size(0))
        return aggr_out
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'

class GNNpool(nn.Module):
    def __init__(self, input_dim, conv_hidden, mlp_hidden, num_clusters, num_layers=2, dropout=0.25):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_layers = num_layers
        
        # GCN layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, conv_hidden))
        self.norms.append(nn.LayerNorm(conv_hidden))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(conv_hidden, conv_hidden))
            self.norms.append(nn.LayerNorm(conv_hidden))
        
        # Residual projection if dimensions don't match
        self.res_proj = None
        if input_dim != conv_hidden:
            self.res_proj = nn.Sequential(
                nn.Linear(input_dim, conv_hidden),
                nn.LayerNorm(conv_hidden)
            )
        
        # Final MLP for clustering
        self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, self.num_clusters)
        )

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        
        # Multi-layer GCN with residual connections
        for i in range(self.num_layers):
            x_prev = x
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            
            if i == 0 and self.res_proj is not None:
                x_prev = self.res_proj(x_prev)
            x = x + x_prev
        
        # Project to cluster assignments
        H = self.mlp(x)
        S = F.softmax(H, dim=1)
        return S

class DeepCutModule(nn.Module):
    def __init__(self, 
                 num_clusters: int, 
                 in_dim: int, 
                 conv_hidden: int = 64,
                 mlp_hidden: int = 64,
                 num_layers: int = 4,
                 smoothness_lambda: float = 0.01,
                 feature_smoothness_lambda: float = 0.09,
                 edge_smoothness_lambda: float = 0.09,
                 use_spatial_pos: bool = True,
                 graph_eps: float = 1e-6,
                 radius_fraction: float = 30.0,
                 normalization: str = 'sym'):
        """
        Enhanced DeepCut implementation with improved stability and additional features.
        
        Args:
            num_clusters: Number of segmentation classes
            in_dim: Input feature dimension
            conv_hidden: Hidden dimension for GCN layers
            mlp_hidden: Hidden dimension for final MLP
            num_layers: Number of GCN layers
            smoothness_lambda: Weight for spatial smoothness loss
            feature_smoothness_lambda: Weight for feature smoothness loss
            edge_smoothness_lambda: Weight for edge awareness in smoothness
            use_spatial_pos: Whether to augment features with spatial positions
            graph_eps: Epsilon for numerical stability
            radius_fraction: Fraction of maximum radius for local connectivity
            normalization: Graph normalization type ('sym' or 'rw')
        """
        super().__init__()
        self.pool = GNNpool(
            input_dim=in_dim + 2 if use_spatial_pos else in_dim,
            conv_hidden=conv_hidden,
            mlp_hidden=mlp_hidden,
            num_clusters=num_clusters,
            num_layers=num_layers
        )
        
        self.num_clusters = num_clusters
        self.smoothness_lambda = smoothness_lambda
        self.feature_smoothness_lambda = feature_smoothness_lambda
        self.edge_smoothness_lambda = edge_smoothness_lambda
        self.use_spatial_pos = use_spatial_pos
        self.graph_eps = graph_eps
        self.radius_fraction = radius_fraction
        self.normalization = normalization
        
        # Spatial smoothness network (depthwise separable convolutions)
        self.smoothness_conv = nn.Sequential(
            nn.Conv2d(num_clusters, num_clusters, 3, padding=1, groups=num_clusters),
            nn.GroupNorm(num_clusters, num_clusters),
            nn.ReLU(),
            nn.Conv2d(num_clusters, num_clusters, 3, padding=1, groups=num_clusters),
            nn.GroupNorm(num_clusters, num_clusters)
        )
        
        # Edge-aware smoothness
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def normalize_adj(self, adj: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """Normalize adjacency matrix based on specified method."""
        if self.normalization == 'sym':
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            return D_inv_sqrt @ adj @ D_inv_sqrt
        elif self.normalization == 'rw':
            deg_inv = deg.pow(-1)
            deg_inv.masked_fill_(deg_inv == float('inf'), 0)
            D_inv = torch.diag(deg_inv)
            return D_inv @ adj
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization}")

    def compute_feature_smoothness(self, features: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Compute feature smoothness loss with edge awareness."""
        B, T, D = features.shape
        
        # Reshape to spatial dimensions (B, D, H, W)
        features_spatial = features.view(B, H, W, D).permute(0, 3, 1, 2)
        
        # Compute edge weights
        edge_weights = torch.sigmoid(self.edge_conv(features_spatial))
        
        # Compute gradients
        grad_y = features_spatial[:, :, 1:, :] - features_spatial[:, :, :-1, :]
        grad_x = features_spatial[:, :, :, 1:] - features_spatial[:, :, :, :-1]
        
        # Weight gradients by edge awareness
        edge_y = edge_weights[:, :, 1:, :]
        edge_x = edge_weights[:, :, :, 1:]
        
        # Compute weighted smoothness losses
        loss_y = (grad_y.pow(2) * edge_y).mean() / D
        loss_x = (grad_x.pow(2) * edge_x).mean() / D
        
        return (loss_y + loss_x) / 2

    def forward(self, token_features: torch.Tensor, k: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DeepCut module.
        
        Args:
            token_features: Input features of shape (B, T, D)
            k: Unused, kept for compatibility
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (Cluster assignments, Loss value)
        """
        B, T, D = token_features.shape
        H = W = int(T ** 0.5)  # Assume square grid
        device = token_features.device
        
        # Create normalized spatial grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        
        # Compute graph radius
        full_radius = torch.sqrt(torch.tensor(2.0, device=device))
        local_radius = full_radius / self.radius_fraction
        
        # Compute feature smoothness loss
        feature_smoothness_loss = self.compute_feature_smoothness(token_features, H, W)
        
        S_list = []
        loss_list = []
        
        for b in range(B):
            feats = token_features[b]
            
            # Augment features with spatial positions if enabled
            if self.use_spatial_pos:
                feats = torch.cat([feats, pos], dim=1)
            
            # Build radius-based graph
            edge_index = radius_graph(x=pos, r=local_radius, loop=True)
            
            # Create sparse adjacency matrix
            adj = SparseTensor(
                row=edge_index[0], 
                col=edge_index[1],
                sparse_sizes=(T, T)
            ).to_dense()
            
            # Compute degree matrix and normalize adjacency
            deg = degree(edge_index[0], num_nodes=T)
            A_tilde = self.normalize_adj(adj, deg)
            
            # Get cluster assignments
            data = Data(x=feats, edge_index=edge_index)
            S = self.pool(data)
            
            # Compute spatial smoothness
            S_spatial = S.transpose(0, 1).view(1, self.num_clusters, H, W)
            S_smooth = self.smoothness_conv(S_spatial)
            smoothness_loss = F.mse_loss(S_smooth, S_spatial)
            
            # Compute DeepCut losses
            S_T = S.transpose(0, 1)
            numerator = torch.trace(S_T @ A_tilde @ S)
            denominator = torch.trace(S_T @ torch.diag(A_tilde.sum(dim=1)) @ S) + self.graph_eps
            loss_c = -numerator / denominator
            
            # Orthogonality loss
            SS = S_T @ S
            I = torch.eye(self.num_clusters, device=device)
            loss_o = torch.norm(SS / (SS.norm() + self.graph_eps) - I / (I.norm() + self.graph_eps), p='fro')
            
            # Combine losses
            loss_b = (
                loss_c + 
                loss_o + 
                self.smoothness_lambda * smoothness_loss
            )
            
            S_list.append(S.unsqueeze(0))
            loss_list.append(loss_b)
        
        # Combine results across batch
        S_all = torch.cat(S_list, dim=0)
        deepcut_loss = (
            torch.stack(loss_list).mean() + 
            self.feature_smoothness_lambda * feature_smoothness_loss
        )
        
        return S_all, deepcut_loss

class VQGPTSegmentation(pl.LightningModule):
    def __init__(self, vqgan_path: str, vqgan_config: dict, gpt_config: dict, 
                 segmentation_config: dict, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Load and freeze pretrained VQGAN
        
        self.vqgan = VQGAN(**vqgan_config)
        checkpoint = torch.load(vqgan_path, map_location=self.device)
        state_dict = {k.replace('vqgan.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.vqgan.load_state_dict(state_dict)
        for param in self.vqgan.parameters():
            param.requires_grad = False
        self.vqgan.eval()

        # Initialize GPT with MLM head
        
        self.gpt = ImageGPT(**gpt_config)
        self.lm_head = nn.Linear(gpt_config['n_embd'], gpt_config['vocab_size'])
        
        # Initialize DeepCut module
        self.deepcut_module = DeepCutModule(
            num_clusters=segmentation_config['num_classes'],
            in_dim=gpt_config['n_embd'],
            conv_hidden=64,
            mlp_hidden=64,
            num_layers=2
        )

        # Loss weights and configurations
        self.mask_prob = gpt_config.get('mask_prob', 0.15)
        self.mask_token_id = gpt_config.get('mask_token_id', 0)
        self.gpt_loss_weight = gpt_config.get('loss_weight', 1.0)
        self.deepcut_loss_weight = segmentation_config.get('deepcut_loss_weight', 0.2)
        
        # Metrics
        self.val_accuracy = Accuracy(task="multiclass", num_classes=segmentation_config['num_classes'])
        self.val_iou = JaccardIndex(task="multiclass", num_classes=segmentation_config['num_classes'])
        self.learning_rate = learning_rate

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
        quant, _, info = self.vqgan.encode(images_norm)
        if isinstance(info, tuple) and len(info) >= 3:
            indices = info[2]
        else:
            indices = info
        B, C, H, W = quant.shape
        indices = indices.view(B, H * W)
        return indices, B, H, W

    def structured_masking(self, indices, mask_prob, patch_size=4):
        B, T = indices.shape
        H = W = int(T ** 0.5)
        indices_2d = indices.view(B, H, W)
        patch_mask = torch.rand(B, H // patch_size, W // patch_size, device=indices.device) < mask_prob
        mask = patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
        masked_indices = indices_2d.clone()
        masked_indices[mask] = self.vqgan.quantize.n_e  # mask token
        return masked_indices.view(B, T), mask.view(B, T)

    def improved_structured_masking(self, indices, mask_prob, strategy='block'):
        B, T = indices.shape
        H = W = int(T ** 0.5)
        indices_2d = indices.view(B, H, W)
        mask = torch.zeros_like(indices_2d, dtype=torch.bool)
        current_epoch = self.trainer.current_epoch
        if strategy == 'block':
            for b in range(B):
                block_h = max(2, min(H // 4, int(H * (0.1 + 0.3 * current_epoch / 100))))
                block_w = max(2, min(W // 4, int(W * (0.1 + 0.3 * current_epoch / 100))))
                area_to_mask = int(H * W * mask_prob)
                num_blocks = max(1, area_to_mask // (block_h * block_w))
                for _ in range(num_blocks):
                    start_h = torch.randint(0, H - block_h + 1, (1,)).item()
                    start_w = torch.randint(0, W - block_w + 1, (1,)).item()
                    mask[b, start_h:start_h+block_h, start_w:start_w+block_w] = True
        elif strategy == 'sliding':
            window_size = 4
            stride = max(1, window_size // 2)
            for b in range(B):
                for i in range(0, H, stride):
                    for j in range(0, W, stride):
                        if torch.rand(1).item() < mask_prob:
                            i_end = min(i + window_size, H)
                            j_end = min(j + window_size, W)
                            mask[b, i:i_end, j:j_end] = True
        elif strategy == 'random_walk':
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for b in range(B):
                num_to_mask = int(H * W * mask_prob)
                masked_count = 0
                while masked_count < num_to_mask:
                    curr_h, curr_w = torch.randint(0, H, (1,)).item(), torch.randint(0, W, (1,)).item()
                    walk_length = min(num_to_mask - masked_count, torch.randint(10, 30, (1,)).item())
                    for _ in range(walk_length):
                        if 0 <= curr_h < H and 0 <= curr_w < W and not mask[b, curr_h, curr_w]:
                            mask[b, curr_h, curr_w] = True
                            masked_count += 1
                        dh, dw = directions[torch.randint(0, 4, (1,)).item()]
                        curr_h, curr_w = curr_h + dh, curr_w + dw
                        curr_h = max(0, min(H - 1, curr_h))
                        curr_w = max(0, min(W - 1, curr_w))
        else:  # 'random'
            mask = torch.rand(B, H, W, device=indices.device) < mask_prob
        masked_indices = indices_2d.clone()
        masked_indices[mask] = self.mask_token_id
        return masked_indices.view(B, T), mask.view(B, T)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        images = images.to(self.device)
        images_norm = 2.0 * images - 1.0

        # Extract VQGAN tokens
        with torch.no_grad():
            indices, B, H, W = self.get_vqgan_output(images_norm)

        # Create masked input for MLM training
        rand = torch.rand(indices.shape, device=indices.device)
        mask = rand < self.mask_prob
        masked_indices = indices.clone()
        masked_indices[mask] = self.mask_token_id

        # Get GPT features and token predictions
        token_features = self.gpt(masked_indices, return_features=True)
        lm_logits = self.lm_head(token_features)

        if self.lm_head is None:
            self.mask_token_id = self.vqgan.quantize.n_e
            current_epoch = self.trainer.current_epoch
            if current_epoch < 10:
                mask_strategy = 'random'
                base_mask_prob = 0.15
            elif current_epoch < 30:
                mask_strategy = 'block'
                base_mask_prob = 0.25
            else:
                mask_strategy = 'sliding'
                base_mask_prob = min(0.35, 0.15 + 0.01 * current_epoch)
            mask_strategy = 'sliding'
            curr_mask_prob = min(0.5, base_mask_prob * (1 + self.global_step / 20000))
            masked_indices, mask = self.improved_structured_masking(indices, curr_mask_prob, strategy=mask_strategy)

        # Compute MLM loss
        lm_loss = F.cross_entropy(lm_logits[mask], indices[mask]) if mask.any() else torch.tensor(0.0, device=self.device)
        
        # DeepCut segmentation
        S_all, deepcut_loss = self.deepcut_module(token_features)
        S_all = S_all.transpose(1, 2).view(B, self.hparams.segmentation_config['num_classes'], H, W)
        
        # Upsample predictions to match target size
        seg_logits = F.interpolate(S_all, size=images.shape[2:], mode='bilinear', align_corners=False)
        
        # Compute segmentation loss
        seg_loss = F.cross_entropy(seg_logits, masks)
        
        # Combined loss
        total_loss = (
            self.gpt_loss_weight * lm_loss + 
            seg_loss + 
            self.deepcut_loss_weight * deepcut_loss
        )

        # Calculate metrics
        with torch.no_grad():
            preds = torch.argmax(seg_logits, dim=1)
            train_dice = self.compute_dice(preds, masks)

        # Logging
        self.log_dict({
            'train/lm_loss': lm_loss,
            'train/seg_loss': seg_loss,
            'train/deepcut_loss': deepcut_loss,
            'train/total_loss': total_loss,
            'train/dice': train_dice,
        }, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        images = images.to(self.device)
        images_norm = 2.0 * images - 1.0

        with torch.no_grad():
            # Get VQGAN tokens
            indices, B, H, W = self.get_vqgan_output(images_norm)

            # Get features and segmentation
            token_features = self.gpt(indices, return_features=True)
            S_all, deepcut_loss = self.deepcut_module(token_features)
            S_all = S_all.transpose(1, 2).view(B, self.hparams.segmentation_config['num_classes'], H, W)
            
            # Upsample predictions
            seg_logits = F.interpolate(S_all, size=images.shape[2:], mode='bilinear', align_corners=False)
            preds = torch.argmax(seg_logits, dim=1)
            
            # Compute metrics
            val_dice = self.compute_dice(preds, masks)
            self.val_accuracy(preds, masks)
            self.val_iou(preds, masks)
            
            # Log metrics
            self.log_dict({
                'val/deepcut_loss': deepcut_loss,
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
        optimizer = torch.optim.AdamW([
            {'params': self.gpt.parameters()},
            {'params': self.lm_head.parameters()},
            {'params': self.deepcut_module.parameters()}
        ], lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
        
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