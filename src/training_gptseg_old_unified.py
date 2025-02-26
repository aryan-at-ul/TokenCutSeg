import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex
from contextlib import contextmanager
import wandb
import numpy as np
# maptplotlit import
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
from PIL import Image
from io import BytesIO

# PyG imports
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph, GCNConv
from torch_geometric.utils import degree
from torch_sparse import SparseTensor

#####################################
# CODEBOOK SEGMENTATION MODULE
#####################################
class CodebookSegmentationModule(nn.Module):
    def __init__(self, codebook_dim: int, num_classes: int, temperature: float = 0.1, use_pos: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.use_pos = use_pos
        in_features = codebook_dim + 2 if use_pos else codebook_dim

        self.feature_projector = nn.Sequential(
            nn.Linear(in_features, codebook_dim),
            nn.LayerNorm(codebook_dim),
            nn.ReLU(),
            nn.Linear(codebook_dim, codebook_dim)
        )
        self.cluster_centroids = nn.Parameter(torch.empty(num_classes, codebook_dim))
        nn.init.xavier_uniform_(self.cluster_centroids)

        self.refinement = nn.Sequential(
            nn.Conv2d(num_classes, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, token_features: torch.Tensor, quant_indices: torch.Tensor):
        B, T, D = token_features.size()
        H = W = int(T ** 0.5)  # assume square layout
        if self.use_pos:
            device = token_features.device
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, 1, H, device=device),
                torch.linspace(0, 1, W, device=device),
                indexing="ij"
            )
            pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (T, 2)
            pos = pos.unsqueeze(0).expand(B, -1, -1)
            token_features = torch.cat([token_features, pos], dim=-1)
        proj = self.feature_projector(token_features)
        proj = F.normalize(proj, dim=-1)
        centroids = F.normalize(self.cluster_centroids, dim=-1)
        similarity = torch.einsum('btd,cd->btc', proj, centroids)
        cluster_logits = similarity / self.temperature
        cluster_probs = F.softmax(cluster_logits, dim=-1)  # (B, T, num_classes)
        # Reshape and refine with convolutions
        cluster_probs_2d = cluster_probs.transpose(1, 2).view(B, self.num_classes, H, W)
        refined_logits = self.refinement(cluster_probs_2d)
        refined_probs = F.softmax(refined_logits, dim=1)
        # Compute losses
        entropy_loss = -(refined_probs * torch.log(refined_probs + 1e-8)).sum(1).mean()
        tv_loss = (torch.abs(refined_probs[:, :, 1:, :] - refined_probs[:, :, :-1, :]).mean() +
                   torch.abs(refined_probs[:, :, :, 1:] - refined_probs[:, :, :, :-1]).mean())
        seg_probs = refined_probs.view(B, self.num_classes, T).transpose(1, 2)
        quant_indices = quant_indices.view(B, T)
        cluster_sim = torch.bmm(seg_probs, seg_probs.transpose(1, 2))
        code_sim = (quant_indices.unsqueeze(2) == quant_indices.unsqueeze(1)).float()
        consistency_loss = F.mse_loss(cluster_sim, code_sim)
        losses = {
            'entropy_loss': entropy_loss,
            'tv_loss': tv_loss,
            'consistency_loss': consistency_loss
        }
        return refined_probs, losses

#####################################
# DIFFERENTIABLE BIPARTITE GRAPH SOLVER
#####################################
class DifferentiableBipartiteGraph(nn.Module):
    def __init__(self, num_classes, feature_dim, nu=4.0, eps=1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.nu = nu
        self.eps = eps
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, feature_dim).div_(feature_dim ** 0.5))
        self.register_buffer('running_lap_loss', torch.zeros(1))
        self.register_buffer('running_entropy_loss', torch.zeros(1))
        self.momentum = 0.9

    def forward(self, token_features):
        B, T, d = token_features.size()
        prototypes = self.class_prototypes.unsqueeze(0).expand(B, self.num_classes, d)
        token_features = F.normalize(token_features, dim=-1, eps=self.eps)
        prototypes = F.normalize(prototypes, dim=-1, eps=self.eps)
        temperature = 0.1
        S = torch.bmm(prototypes, token_features.transpose(1, 2)) / temperature
        B_assign = F.softmax(S, dim=2)
        D_r = torch.diag_embed(B_assign.sum(dim=2) + self.eps)
        D_q = torch.diag_embed(B_assign.sum(dim=1) + self.eps)
        B_assign_T = B_assign.transpose(1, 2)
        top = torch.cat([D_r, -B_assign], dim=2)
        bottom = torch.cat([-B_assign_T, D_q], dim=2)
        L_full = torch.cat([top, bottom], dim=1)
        I_full = torch.eye(L_full.size(1), device=L_full.device, dtype=L_full.dtype).unsqueeze(0).expand(B, -1, -1)
        L_reg = L_full + self.eps * I_full
        eigenvals = torch.linalg.eigvals(L_reg)
        condition_number = torch.abs(eigenvals).max(dim=1)[0] / (torch.abs(eigenvals).min(dim=1)[0] + self.eps)
        sign, logdet = torch.slogdet(L_reg)
        lap_loss = -logdet.mean()
        B_assign_safe = torch.clamp(B_assign, min=1e-8)
        entropy = - (B_assign_safe * torch.log(B_assign_safe)).sum(dim=2)
        heavy_tail_loss = entropy.mean()
        with torch.no_grad():
            self.running_lap_loss.mul_(self.momentum).add_(lap_loss.detach() * (1 - self.momentum))
            self.running_entropy_loss.mul_(self.momentum).add_(heavy_tail_loss.detach() * (1 - self.momentum))
        lap_scale = 1.0 / (self.running_lap_loss + 1.0)
        entropy_scale = 1.0 / (self.running_entropy_loss + 1.0)
        bip_loss = lap_scale * lap_loss + self.nu * entropy_scale * heavy_tail_loss
        stats = {'condition_number': condition_number.mean(), 'lap_loss': lap_loss, 'entropy_loss': heavy_tail_loss}
        return B_assign, S, bip_loss, stats



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, knn_graph, radius_graph
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_sparse import SparseTensor

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

##############################################
# GNN Pool: Now Outputs Node Features Instead of Cluster Prediction
##############################################
class GNNpool(nn.Module):
    def __init__(self, input_dim, conv_hidden, num_layers=3, 
                 batch_norm=True, cat=True, dropout=0.15):
        """
        Args:
            input_dim (int): Input feature dimension.
            conv_hidden (int): Hidden dimension for intermediate features.
            num_layers (int): Number of RelConv layers.
            batch_norm (bool): Whether to use layer normalization.
            cat (bool): Whether to concatenate features from all layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        cur_dim = input_dim
        for i in range(num_layers):
            self.convs.append(RelConv(cur_dim, conv_hidden))
            if self.batch_norm:
                self.batch_norms.append(nn.LayerNorm(conv_hidden))
            cur_dim = conv_hidden  # For next layer
        
        # Final embedding dimension: either concatenated from all layers (plus the input) or just last layer.
        final_dim = input_dim + num_layers * conv_hidden if self.cat else conv_hidden
            
        self.final = nn.Linear(final_dim, conv_hidden)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_normal_(self.final.weight, gain=0.01)
        if self.final.bias is not None:
            nn.init.zeros_(self.final.bias)
        
    def reset_parameters(self):
        for i, conv in enumerate(self.convs):
            conv.reset_parameters()
            if self.batch_norm:
                self.batch_norms[i].reset_parameters()
        self.final.reset_parameters()
        
    def forward(self, data: Data):
        """
        Args:
            data (Data): Contains x (node features) and edge_index.
        Returns:
            f (torch.Tensor): Node features of shape [num_nodes, conv_hidden].
        """
        x, edge_index = data.x, data.edge_index
        xs = [x]
        for i, conv in enumerate(self.convs):
            x = conv(xs[-1], edge_index)
            x = F.relu(x)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        if self.cat:
            x_cat = torch.cat(xs, dim=-1)
        else:
            x_cat = xs[-1]
        f = self.final(x_cat)
        return f


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.act1 = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act2 = nn.SiLU()
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual  # Actual residual connection
        x = self.act2(x)
        return x


##############################################
# DEEPCUT MODULE: Final Conv Head for n-Cluster Prediction
##############################################
class DeepCutModule(nn.Module):
    def __init__(self, 
                 num_clusters: int, 
                 in_dim: int, 
                 conv_hidden: int = 128,
                 mlp_hidden: int = 128,
                 num_layers: int = 3,
                 smoothness_lambda: float = 0.01,
                 feature_smoothness_lambda: float = 0.05,
                 edge_smoothness_lambda: float = 0.05,
                 use_spatial_pos: bool = True,
                 graph_eps: float = 1e-6,
                 radius_fraction: float = 30.0,
                 normalization: str = 'sym',
                 use_multi_scale: bool = False):
        super().__init__()
        self.num_clusters = num_clusters
        self.smoothness_lambda = smoothness_lambda
        self.feature_smoothness_lambda = feature_smoothness_lambda
        self.edge_smoothness_lambda = edge_smoothness_lambda
        self.use_spatial_pos = use_spatial_pos
        self.graph_eps = graph_eps
        self.radius_fraction = radius_fraction
        self.normalization = normalization
        self.use_multi_scale = use_multi_scale
        
        # GNN Pool now returns node features.
        self.pool = GNNpool(
            input_dim=in_dim + 2 if use_spatial_pos else in_dim,
            conv_hidden=conv_hidden,
            num_layers=num_layers
        )
        # Final convolutional head to predict n clusters.
        # This head works on the spatially reshaped features.
        # self.final_conv = nn.Sequential(
        #     nn.Conv2d(conv_hidden, conv_hidden, kernel_size=3, padding=1),
        #     nn.GroupNorm(8, conv_hidden),
        #     nn.ReLU(),
        #     nn.Conv2d(conv_hidden, num_clusters, kernel_size=1)
        # )
        
        self.final_conv = nn.Sequential(
            ResidualConvBlock(conv_hidden),
            # Final 1x1 projection to output classes
            nn.Conv2d(conv_hidden, num_clusters, kernel_size=1)
        )


        # Additional modules (for smoothness, boundary refinement, etc.)
        self.smoothness_conv = nn.Sequential(
            nn.Conv2d(num_clusters, num_clusters, kernel_size=3, padding=1, dilation=1, groups=num_clusters),
            nn.GroupNorm(num_clusters, num_clusters),
            nn.ReLU(),
            nn.Conv2d(num_clusters, num_clusters, kernel_size=3, padding=2, dilation=2, groups=num_clusters),
            nn.GroupNorm(num_clusters, num_clusters),
            nn.ReLU(),
            nn.Conv2d(num_clusters, num_clusters, kernel_size=3, padding=1, groups=num_clusters),
            nn.GroupNorm(num_clusters, num_clusters)
        )
        
        in_channels_edge = in_dim + 2 if self.use_spatial_pos else in_dim
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels_edge, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        
        self.boundary_refinement = nn.Sequential(
            nn.Conv2d(num_clusters + in_dim, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, num_clusters, kernel_size=1)
        )
        
        if self.use_multi_scale:
            self.scale_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_dim, conv_hidden, kernel_size=ks, padding=ks//2),
                    nn.GroupNorm(8, conv_hidden),
                    nn.ReLU(),
                    nn.Conv2d(conv_hidden, in_dim, kernel_size=1)
                ) for ks in [3, 5, 7]
            ])
            self.scale_weights = nn.Parameter(torch.ones(len(self.scale_convs) + 1))
    
    def normalize_adj(self, adj: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
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
        B, T, D = features.shape
        features_spatial = features.view(B, H, W, D).permute(0, 3, 1, 2)
        edge_weights = torch.sigmoid(self.edge_conv(features_spatial))
        grad_y = features_spatial[:, :, 1:, :] - features_spatial[:, :, :-1, :]
        grad_x = features_spatial[:, :, :, 1:] - features_spatial[:, :, :, :-1]
        loss_y = (grad_y.pow(2) * edge_weights[:, :, 1:, :]).mean() / D
        loss_x = (grad_x.pow(2) * edge_weights[:, :, :, 1:]).mean() / D
        return (loss_y + loss_x) / 2
    
    def apply_multi_scale_processing(self, features_spatial):
        feature_scales = [features_spatial]
        for conv in self.scale_convs:
            feature_scales.append(conv(features_spatial))
        scale_weights = F.softmax(self.scale_weights, dim=0)
        multi_scale_features = sum(w * fs for w, fs in zip(scale_weights, feature_scales))
        return multi_scale_features

    def forward(self, token_features: torch.Tensor, k: int = 8):
        """
        Args:
            token_features (torch.Tensor): [B, T, D] tokens. Assumes a square grid (H x W) with H = W = sqrt(T).
            k (int): Not used directly here (left for compatibility).
        Returns:
            final_S (torch.Tensor): [B, num_clusters, H, W] segmentation predictions.
            deepcut_loss (torch.Tensor): Combined DeepCut loss.
        """
        B, T, D = token_features.shape
        H = W = int(T ** 0.5)
        device = token_features.device
        
        # Optionally apply multi-scale enhancement
        if hasattr(self, 'use_multi_scale') and self.use_multi_scale:
            features_spatial = token_features.view(B, H, W, D).permute(0, 3, 1, 2)
            multi_scale_features = self.apply_multi_scale_processing(features_spatial)
            token_features_enhanced = multi_scale_features.permute(0, 2, 3, 1).reshape(B, T, D)
        else:
            token_features_enhanced = token_features
        
        # Create positional grid and append if needed.
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        if self.use_spatial_pos:
            # Concatenate position to token features.
            token_features_enhanced = torch.cat([token_features_enhanced, pos.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        
        # Compute feature smoothness loss.
        feature_smoothness_loss = self.compute_feature_smoothness(token_features_enhanced, H, W)
        
        S_list = []
        loss_list = []
        refined_S_list = []
        for b in range(B):
            feats = token_features_enhanced[b]  # [T, feature_dim]
            # Build graph using spatial positions.
            # edge_index = radius_graph(x=pos, r=(torch.sqrt(torch.tensor(2.0, device=device)) / self.radius_fraction), loop=True)
            edge_index = knn_graph(pos, k=8, loop=True)
            # Create a sparse adjacency matrix.
            adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                               sparse_sizes=(T, T)).to_dense()
            deg = degree(edge_index[0], num_nodes=T)
            A_tilde = self.normalize_adj(adj, deg)
            
            # Build a Data object for the GNN pool.
            data = Data(x=feats, edge_index=edge_index)
            # Get node features from the GNN pool.
            f = self.pool(data)  # [T, conv_hidden]
            # Reshape features into spatial grid.
            f_spatial = f.view(H, W, -1).permute(2, 0, 1).unsqueeze(0)  # [1, conv_hidden, H, W]
            
            # Final convolutional head to predict clusters.
            S = self.final_conv(f_spatial)  # [1, num_clusters, H, W]
            # Optionally refine smoothness.
            S_smooth = self.smoothness_conv(S)
            smoothness_loss = F.mse_loss(S_smooth, S)
            # Boundary refinement branch.
            orig_feats_spatial = token_features[b].view(H, W, D).permute(2, 0, 1).unsqueeze(0)
            refinement_input = torch.cat([S, orig_feats_spatial], dim=1)
            refined_S = self.boundary_refinement(refinement_input)
            refined_S = F.softmax(refined_S, dim=1)
            # Graph cut loss (using dense adj).
            S_flat = S.view(self.num_clusters, T).transpose(0, 1)  # [T, num_clusters]
            numerator = torch.trace(S_flat.transpose(0,1) @ A_tilde @ S_flat)
            denominator = torch.trace(S_flat.transpose(0,1) @ torch.diag(A_tilde.sum(dim=1)) @ S_flat) + self.graph_eps
            loss_c = -numerator / denominator
            # Orthogonality loss on cluster assignments.
            SS = S_flat.transpose(0,1) @ S_flat
            I = torch.eye(self.num_clusters, device=device)
            loss_o = torch.norm(SS / (SS.norm() + self.graph_eps) - I / (I.norm() + self.graph_eps), p='fro')
            # Boundary loss via KL divergence.
            boundary_loss = F.kl_div(F.log_softmax(refined_S, dim=1),
                                     F.softmax(S, dim=1),
                                     reduction='batchmean')
            loss_b = loss_c + loss_o + self.smoothness_lambda * smoothness_loss + 0.01 * boundary_loss
            loss_list.append(loss_b)
            S_list.append(S)
            refined_S_list.append(refined_S)
        
        S_all = torch.cat(S_list, dim=0)  # [B, num_clusters, H, W]
        refined_S_all = torch.cat(refined_S_list, dim=0)  # [B, num_clusters, H, W]
        final_S = refined_S_all
        deepcut_loss = torch.stack(loss_list).mean() + self.feature_smoothness_lambda * feature_smoothness_loss
        return final_S, deepcut_loss




#####################################
# UTILITY FUNCTIONS
#####################################
def compute_iou(pred, target, smooth=1e-6):
    pred_inds = (pred == 1)
    target_inds = (target == 1)
    intersection = (pred_inds & target_inds).float().sum()
    union = (pred_inds | target_inds).float().sum()
    return (intersection + smooth) / (union + smooth)

def compute_dice(pred, target, num_classes=2, smooth=1e-6):
    dice_scores = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).float().sum()
        dice_score = (2 * intersection + smooth) / (pred_inds.float().sum() + target_inds.float().sum() + smooth)
        dice_scores.append(dice_score)
    return sum(dice_scores) / len(dice_scores)

def compute_accuracy(pred, target):
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    return correct / total

#####################################
# FUSION SEGMENTATION HEAD
#####################################
class FusionSegmentationHead(nn.Module):
    def __init__(self, gpt_dim, quant_dim, num_classes):
        super(FusionSegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(gpt_dim + quant_dim, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, gpt_features, quant_features, target_size):
        B, T, gpt_dim = gpt_features.shape
        H_q, W_q = quant_features.shape[2], quant_features.shape[3]
        gpt_features_spatial = gpt_features.transpose(1, 2).view(B, gpt_dim, H_q, W_q)
        fused = torch.cat([gpt_features_spatial, quant_features], dim=1)
        x = self.conv1(fused)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        seg_logits = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return seg_logits

#####################################
# INTEGRATED VQGAN+GPT+SEGMENTATION MODEL
#####################################



import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex
import math
from contextlib import contextmanager
import wandb



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from contextlib import contextmanager
import wandb

from torchmetrics import Accuracy, JaccardIndex

# Custom modules – adjust the import paths as needed
from src.vqa.vqagan import VQGAN
from src.imagegpt_model2 import ImageGPT



def truncated_normal_(tensor, mean=0.0, std=0.02, a=-0.04, b=0.04):
    """
    Fills the input tensor with values drawn from a truncated normal distribution.
    """
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    return tensor


def compute_iou(pred, target, num_classes=None):
    """
    Compute the Intersection over Union (IoU) for a single sample.
    """
    if num_classes is None:
        num_classes = int(max(pred.max().item(), target.max().item()) + 1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()
        if union == 0:
            ious.append(torch.tensor(1.0, device=pred.device))
        else:
            ious.append(intersection / union)
    return torch.stack(ious).mean()


def compute_accuracy(pred, target):
    """
    Compute the pixel-wise accuracy for a single sample.
    """
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return correct / total


class MaskPredictionHead(nn.Module):
    """
    Head for predicting semantic segmentation masks from GPT features.
    """
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        use_attention_features: bool = True,
        dropout: float = 0.1,
        initializer_range: float = 0.02
    ):
        super().__init__()
        self.use_attention_features = use_attention_features

        # Feature transformation layers
        self.feature_transform = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        if use_attention_features:
            # Attention-based features transformation
            self.attention_transform = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            # Combined projection from concatenated features
            self.combined_projection = nn.Linear(hidden_size, num_classes)
        else:
            # Direct projection using only token features
            self.mask_projection = nn.Linear(hidden_size // 2, num_classes)

        self._init_weights(initializer_range)

    def _init_weights(self, initializer_range: float):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                truncated_normal_(module.weight, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, hidden_states, attention_maps=None):
        """
        Args:
            hidden_states: Tensor of shape [B, T, hidden_size]
            attention_maps: Optional tensor of shape [B, num_heads, T, T]
        
        Returns:
            Logits with shape [B, T, num_classes]
        """
        token_features = self.feature_transform(hidden_states)

        if self.use_attention_features and attention_maps is not None:
            # Average attention across heads
            avg_attention = attention_maps.mean(dim=1)  # [B, T, T]
            # Compute attention-weighted features
            attention_features = torch.bmm(avg_attention, hidden_states)
            attention_features = self.attention_transform(attention_features)
            # Concatenate token and attention features
            combined_features = torch.cat([token_features, attention_features], dim=-1)
            logits = self.combined_projection(combined_features)
        else:
            logits = self.mask_projection(token_features)

        return logits


    def get_spatial_logits(self, hidden_states, attention_maps=None, height=None, width=None):
        """
        Rearrange token logits into spatial format.
        """
        B, T, _ = hidden_states.shape

        logits = self.forward(hidden_states, attention_maps)  # [B, T, num_classes]

        if height is None:
            height = int(math.sqrt(T))
            width = height
        elif width is None:
            width = T // height

        assert height * width == T, f"Token count {T} doesn't match spatial dims {height}x{width}"

        logits = logits.permute(0, 2, 1)  # [B, num_classes, T]
        logits = logits.view(B, -1, height, width)  # [B, num_classes, H, W]

        return logits


class VQGPTSegmentation(pl.LightningModule):
    def __init__(self, vqgan_config: dict, gpt_config: dict, segmentation_config: dict, learning_rate: float = 1e-4,
                 gradient_accumulation_steps: int = 2):
        super().__init__()
        self.save_hyperparameters()  # Saves all init arguments
        self.hparams.gpt_loss_weight = gpt_config.get("loss_weight", 1.0)
        self.sup_fraction = segmentation_config.get("sup_fraction", 1.0)
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Initialize VQGAN and ImageGPT
        self.vqgan = VQGAN(**vqgan_config)
    # Set mask token based on codebook size
        vqgan_vocab_size = vqgan_config.get('n_embed', 1024)
        mask_token_id = vqgan_vocab_size  # Use codebook size as mask token
        gpt_config.update({
            'mask_token_id': mask_token_id,
            'use_gated_mlp': True,
            'use_flash_attn': True,
            'position_embedding_type': 'absolute',
            'output_attentions': True  # Enable attention output
        })
        self.gpt = ImageGPT(**gpt_config)
        if hasattr(self.gpt.transformer, 'mlm_layer'):
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(gpt_config['n_embd'], gpt_config['vocab_size'])

        # Calculate expected token dimensions based on downsampling factor
        downsampling_factor = 4  # VQGAN typically downsamples by 4x
        input_size = segmentation_config.get('grid_size', (224, 224))
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        
        token_h, token_w = input_size[0] // downsampling_factor, input_size[1] // downsampling_factor
        self.expected_token_shape = (token_h, token_w)
        self.expected_token_count = token_h * token_w
        
        print(f"Expected token shape: {self.expected_token_shape}, count: {self.expected_token_count}")

        # Segmentation modules
        self.use_codebook = segmentation_config.get("use_codebook", False)
        self.use_deepcut = segmentation_config.get("use_deepcut", False)
        self.use_bipartite = segmentation_config.get("use_bipartite", False)
        self.use_gpt_mask_prediction = segmentation_config.get("use_gpt_mask_prediction", True)
        self.use_attn_supervision = segmentation_config.get("use_attn_supervision", False)

        self.codebook_segmentation = CodebookSegmentationModule(
            codebook_dim=gpt_config['n_embd'],
            num_classes=segmentation_config['num_classes'],
            temperature=0.1,
            use_pos=True
        ) if self.use_codebook else None

        self.deepcut_module = DeepCutModule(
            num_clusters=segmentation_config['num_classes'],
            in_dim=gpt_config['n_embd'],
            conv_hidden=64,
            mlp_hidden=64,
            num_layers=3
        ) if self.use_deepcut else None

        self.diff_bipartite = DifferentiableBipartiteGraph(
            num_classes=segmentation_config['num_classes'],
            feature_dim=gpt_config['n_embd'],
            nu=segmentation_config.get('nu', 4.0),
            eps=segmentation_config.get('eps', 1e-3)
        ) if self.use_bipartite else None




        # GPT Mask Prediction Module
        if self.use_gpt_mask_prediction:
            self.mask_prediction_head = MaskPredictionHead(
                hidden_size=gpt_config['n_embd'],
                num_classes=segmentation_config['num_classes'],
                use_attention_features=True,
                dropout=0.1
            )

        # Attention-based Segmentation Module - using a lazy module approach
        if self.use_attn_supervision:
            self.attn_seg_hidden_dim = 512
            self.attn_seg_num_classes = segmentation_config['num_classes']
            
            # Create a function that will build the module with correct dimensions when needed
            def build_attn_seg_module(token_count):
                return nn.Sequential(
                    nn.Linear(token_count, self.attn_seg_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.attn_seg_hidden_dim, self.attn_seg_hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(self.attn_seg_hidden_dim // 2, self.attn_seg_num_classes)
                )
            
            self.build_attn_seg_module = build_attn_seg_module
            self.attn_segmentation_module = None  # Will be created in first forward pass




        # Learnable blending parameters
        num_branches = (1 + int(self.use_codebook) + int(self.use_deepcut) +
                        int(self.use_bipartite) + int(self.use_gpt_mask_prediction) +
                        int(self.use_attn_supervision))
        self.blend_alpha = nn.Parameter(torch.zeros(1))
        self.blend_params = nn.Parameter(torch.zeros(num_branches))

        # Loss weights
        self.codebook_weight = vqgan_config.get('codebook_weight', 0.1)
        self.deepcut_loss_weight = segmentation_config.get('deepcut_loss_weight', 0.2)
        self.bip_loss_weight = segmentation_config.get('bip_loss_weight', 0.2)
        self.mask_pred_loss_weight = segmentation_config.get('mask_pred_loss_weight', 0.5)
        self.attn_seg_loss_weight = segmentation_config.get('attn_seg_loss_weight', 0.3)
        self.dice_loss_weight = segmentation_config.get("dice_loss_weight", 1.0)

        # Evaluation metrics
        self.val_accuracy = Accuracy(task="multiclass", num_classes=segmentation_config['num_classes'], average='macro')
        self.val_iou = JaccardIndex(task="multiclass", num_classes=segmentation_config['num_classes'], average='macro')
        self.mask_pred_accuracy = Accuracy(task="multiclass", num_classes=segmentation_config['num_classes'], average='macro')
        self.mask_pred_iou = JaccardIndex(task="multiclass", num_classes=segmentation_config['num_classes'], average='macro')

        self.learning_rate = learning_rate
        self.automatic_optimization = False

        # Track gradient accumulation steps as a buffer
        self.register_buffer('accumulated_steps', torch.tensor(0))

    # -----------------------------
    # Utility methods
    # -----------------------------
    @contextmanager
    def ema_scope(self):
        if hasattr(self.vqgan.quantize, "embedding_ema"):
            old_val = self.vqgan.quantize.embedding_ema
            self.vqgan.quantize.embedding_ema = True
        else:
            old_val = None
        try:
            yield
        finally:
            if old_val is not None:
                self.vqgan.quantize.embedding_ema = old_val

    def clip_gradients(self, optimizer):
        for param_group in optimizer.param_groups:
            group_name = param_group.get('name', '')
            if 'vqgan' in group_name or 'discriminator' in group_name:
                torch.nn.utils.clip_grad_norm_(param_group['params'], 1.0)
            elif 'gpt' in group_name:
                torch.nn.utils.clip_grad_norm_(param_group['params'], 0.5)
            else:
                torch.nn.utils.clip_grad_norm_(param_group['params'], 1.0)

    def compute_token_weights(self, indices, vocab_size):
        with torch.no_grad():
            token_counts = torch.bincount(indices.flatten(), minlength=vocab_size)
            token_freq = token_counts.float() / token_counts.sum()
            token_weights = 1.0 / (token_freq + 1e-5)
            token_weights = token_weights / token_weights.mean()
            token_weights = torch.clamp(token_weights, 0.1, 10.0)
        return token_weights

    def dice_loss_fn(self, inputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        num_classes = self.hparams.segmentation_config['num_classes']
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        class_counts = targets_one_hot.sum(dim=(0, 2, 3)) + eps
        total_pixels = targets_one_hot.shape[0] * targets_one_hot.shape[2] * targets_one_hot.shape[3]
        class_weights = (total_pixels / (class_counts * num_classes)).clamp(0.5, 2.0)
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        total = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_scores = (2 * intersection + eps) / (total + eps)
        weighted_dice_loss = (1 - dice_scores) * class_weights.unsqueeze(0)
        return weighted_dice_loss.mean()

    def compute_dice(self, preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        num_classes = self.hparams.segmentation_config['num_classes']
        batch_size = preds.shape[0]
        dice_scores = torch.zeros(batch_size, num_classes, device=preds.device)
        for b in range(batch_size):
            for c in range(num_classes):
                pred_c = (preds[b] == c).float()
                target_c = (targets[b] == c).float()
                intersection = (pred_c * target_c).sum()
                cardinality = pred_c.sum() + target_c.sum()
                if cardinality < eps:
                    dice_scores[b, c] = 1.0 if (pred_c.sum() < eps and target_c.sum() < eps) else 0.0
                else:
                    dice_scores[b, c] = (2 * intersection + eps) / (cardinality + eps)
        return dice_scores.mean()

    # -----------------------------
    # Masking methods
    # -----------------------------
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



    def _ensure_attn_module(self, token_count):
        """Ensure the attention segmentation module exists with correct dimensions"""
        if self.attn_segmentation_module is None:
            print(f"Creating attention segmentation module with input dim: {token_count}")
            self.attn_segmentation_module = self.build_attn_seg_module(token_count).to(self.device)
        elif self.attn_segmentation_module[0].in_features != token_count:
            print(f"Rebuilding attention segmentation module - expected: {self.expected_token_count}, " 
                f"got: {token_count}")
            self.attn_segmentation_module = self.build_attn_seg_module(token_count).to(self.device)




    # -----------------------------
    # Training Step
    # -----------------------------
    def training_step(self, batch, batch_idx):
        images, masks = batch
        images = images.to(self.device)
        masks = masks.to(self.device)
        opt = self.optimizers()

        images_norm = 2.0 * images - 1.0

        # VQGAN training
        with self.ema_scope():
            loss_vq, vq_log_dict = self.vqgan(images_norm, optimizer_idx=0, global_step=self.global_step, seg_mask=masks)
            scaled_loss_vq = loss_vq / self.gradient_accumulation_steps
            self.manual_backward(scaled_loss_vq)
            loss_disc = None
            if self.global_step >= self.vqgan.loss.disc_start:
                steps_since_disc_start = self.global_step - self.vqgan.loss.disc_start
                disc_weight = min(1.0, 0.01 * ((steps_since_disc_start // 1000) + 1))
                loss_disc, disc_log_dict = self.vqgan(images_norm, optimizer_idx=1, global_step=self.global_step)
                if loss_disc is not None:
                    scaled_loss_disc = disc_weight * loss_disc / self.gradient_accumulation_steps
                    self.manual_backward(scaled_loss_disc)

        # GPT + Segmentation training
        with torch.no_grad():
            quant, codebook_loss, info = self.vqgan.encode(images_norm, masks)
            quant = quant.detach()
            indices = info[2] if isinstance(info, tuple) and len(info) >= 3 else info
            indices = indices.detach()
        B, C, H_q, W_q = quant.shape
        T = H_q * W_q
        indices = indices.view(B, T)

        ds_masks = F.interpolate(masks.unsqueeze(1).float(), size=(H_q, W_q), mode='nearest').long().squeeze(1)

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
        curr_mask_prob = min(0.5, base_mask_prob * (1 + self.global_step / 20000))
        masked_indices, mask = self.improved_structured_masking(indices, curr_mask_prob, strategy=mask_strategy)

        if self.lm_head is None:
            token_features, attn_outputs = self.gpt(masked_indices, return_features=True, return_attn=True)
            lm_logits = self.gpt.transformer.mlm_layer(
                token_features,
                self.gpt.transformer.embed.word_embeddings.weight
            )
        else:
            token_features, attn_outputs = self.gpt(masked_indices, return_features=True, return_attn=True)
            lm_logits = self.lm_head(token_features)

        # MLM loss with importance weighting
        if mask.any():
            token_weights = self.compute_token_weights(indices, self.vqgan.quantize.n_e + 1)
            mask_weights = token_weights[indices[mask]].to(self.device)
            lm_loss = F.cross_entropy(lm_logits[mask], indices[mask], reduction='none')
            lm_loss = (lm_loss * mask_weights).mean()
        else:
            lm_loss = torch.tensor(0.0, device=self.device)

        mask_pred_loss = torch.tensor(0.0, device=self.device)
        mask_pred_upsampled = None

        if self.use_gpt_mask_prediction:
            last_layer_attn = attn_outputs[-1] if attn_outputs else None
            mask_pred_logits = self.mask_prediction_head.get_spatial_logits(
                token_features,
                attention_maps=last_layer_attn,
                height=H_q,
                width=W_q
            )
            # mask_pred_logits = self.mask_prediction_head(token_features, last_layer_attn, H_q, W_q)
            token_mask_ce_loss = F.cross_entropy(mask_pred_logits, ds_masks)
            token_mask_dice_loss = self.dice_loss_fn(mask_pred_logits, ds_masks)
            token_mask_pred_loss = token_mask_ce_loss + self.dice_loss_weight * token_mask_dice_loss
            mask_pred_upsampled = F.interpolate(
                mask_pred_logits,
                size=images.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            mask_ce_loss = F.cross_entropy(mask_pred_upsampled, masks)
            mask_dice_loss = self.dice_loss_fn(mask_pred_upsampled, masks)
            mask_pred_loss = token_mask_pred_loss + mask_ce_loss + self.dice_loss_weight * mask_dice_loss

        attn_seg_loss = torch.tensor(0.0, device=self.device)
        attn_seg_logits = None

        if self.use_attn_supervision and attn_outputs:
            # Process attention from last layer
            last_layer_attn = attn_outputs[-1]  # Shape: [B, num_heads, T, T]
            avg_attn = last_layer_attn.mean(dim=1)  # Average across heads
            
            # Extract attention patterns for segmentation
            B, T, _ = avg_attn.shape  # Get actual token count
            attn_features = avg_attn.view(B, T, T).permute(0, 2, 1)  # Shape: [B, T, T]
            
            # Ensure module exists with correct dimensions
            self._ensure_attn_module(T)
                
            attn_seg_token_logits = self.attn_segmentation_module(attn_features)  # [B, T, num_classes]

            attn_seg_token_logits = attn_seg_token_logits.permute(0, 2, 1).view(B, -1, H_q, W_q)
            token_attn_ce_loss = F.cross_entropy(attn_seg_token_logits, ds_masks)
            attn_seg_logits = F.interpolate(
                attn_seg_token_logits,
                size=images.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            attn_ce_loss = F.cross_entropy(attn_seg_logits, masks)
            attn_dice_loss = self.dice_loss_fn(attn_seg_logits, masks)
            attn_seg_loss = token_attn_ce_loss + attn_ce_loss + self.dice_loss_weight * attn_dice_loss

        seg_loss = 0.0
        if self.use_deepcut:
            S_all, deepcut_loss = self.deepcut_module(token_features, k=8)
            deepcut_logits = F.interpolate(S_all, size=images.shape[2:], mode='bilinear', align_corners=False)
            seg_loss += self.deepcut_loss_weight * deepcut_loss
        else:
            deepcut_logits = None
            deepcut_loss = torch.tensor(0.0, device=self.device)

        seg_outputs = []
        if self.use_deepcut and deepcut_logits is not None:
            seg_outputs.append(deepcut_logits)
        if self.use_gpt_mask_prediction and mask_pred_upsampled is not None:
            seg_outputs.append(mask_pred_upsampled)
        if self.use_attn_supervision and attn_seg_logits is not None:
            seg_outputs.append(attn_seg_logits)

        if not seg_outputs:
            final_seg_logits = torch.zeros((B, self.hparams.segmentation_config['num_classes'],
                                            images.shape[2], images.shape[3]), device=self.device)
        elif len(seg_outputs) == 1:
            final_seg_logits = seg_outputs[0]
        else:
            if len(self.blend_params) != len(seg_outputs):
                self.blend_params = nn.Parameter(torch.zeros(len(seg_outputs), device=self.device))
            blend_weights = F.softmax(self.blend_params, dim=0)
            final_seg_logits = sum(w * output for w, output in zip(blend_weights, seg_outputs))

        sup_ce_loss = F.cross_entropy(final_seg_logits, masks)
        sup_dice_loss = self.dice_loss_fn(final_seg_logits, masks)
        sup_loss = sup_ce_loss + self.dice_loss_weight * sup_dice_loss

        diversity_loss = torch.tensor(0.0, device=self.device)
        if len(seg_outputs) >= 2:
            for i in range(len(seg_outputs)):
                for j in range(i+1, len(seg_outputs)):
                    diversity_loss += 0.1 * (1.0 - F.cosine_similarity(
                        seg_outputs[i].view(B, -1),
                        seg_outputs[j].view(B, -1)
                    ).mean())

        total_loss = (
            self.hparams.gpt_loss_weight * lm_loss +
            self.mask_pred_loss_weight * mask_pred_loss +
            self.attn_seg_loss_weight * attn_seg_loss +
            seg_loss + sup_loss + 0.05 * diversity_loss
        )

        scaled_total_loss = total_loss / self.gradient_accumulation_steps
        self.manual_backward(scaled_total_loss)
        self.accumulated_steps += 1

        if self.accumulated_steps >= self.gradient_accumulation_steps:
            self.clip_gradients(opt)
            opt.step()
            opt.zero_grad()
            self.accumulated_steps.zero_()

        train_preds = torch.argmax(final_seg_logits, dim=1)
        train_dice = self.compute_dice(train_preds, masks)
        custom_ious = [compute_iou(train_preds[i], masks[i]) for i in range(train_preds.size(0))]
        custom_accs = [compute_accuracy(train_preds[i], masks[i]) for i in range(train_preds.size(0))]
        train_custom_iou = torch.stack(custom_ious).mean()
        train_custom_acc = torch.stack(custom_accs).mean()

        if mask.any():
            token_preds = torch.argmax(lm_logits[mask], dim=1)
            token_acc = (token_preds == indices[mask]).float().mean()
        else:
            token_acc = torch.tensor(1.0, device=self.device)

        log_dict = {
            'train/loss_vq': loss_vq,
            'train/loss_disc': loss_disc if loss_disc is not None else 0.0,
            'train/lm_loss': lm_loss,
            'train/lm_accuracy': token_acc,
            'train/mask_pred_loss': mask_pred_loss if self.use_gpt_mask_prediction else 0.0,
            'train/attn_seg_loss': attn_seg_loss if self.use_attn_supervision else 0.0,
            'train/deepcut_loss': deepcut_loss if self.use_deepcut else 0.0,
            'train/sup_ce_loss': sup_ce_loss,
            'train/sup_dice_loss': sup_dice_loss,
            'train/diversity_loss': diversity_loss,
            'train/total_loss': total_loss,
            'train/dice': train_dice,
            'train/custom_iou': train_custom_iou,
            'train/custom_acc': train_custom_acc,
            'train/mask_probability': curr_mask_prob,
            'train/accumulated_steps': self.accumulated_steps.item(),
        }

        log_dict.update({f'train/blend_weight_{i}': w.item() 
                         for i, w in enumerate(F.softmax(self.blend_params, dim=0))})
        if vq_log_dict is not None:
            for k, v in vq_log_dict.items():
                log_dict[f'train/vq/{k}'] = v

        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True)
        return total_loss

    # -----------------------------
    # Validation Step
    # -----------------------------
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        images = images.to(self.device)
        masks = masks.to(self.device)
        images_norm = 2.0 * images - 1.0

        with torch.no_grad(), self.ema_scope():
            recon, codebook_loss, info = self.vqgan(images_norm, seg_mask=masks)
            indices = info[2] if isinstance(info, tuple) and len(info) >= 3 else info

            quant, _, _ = self.vqgan.encode(images_norm, masks)
            quant = quant.detach()
            B, C, H_q, W_q = quant.shape
            T = H_q * W_q
            indices = indices.view(B, T)

            ds_masks = F.interpolate(masks.unsqueeze(1).float(), size=(H_q, W_q), mode='nearest').long().squeeze(1)

            self.mask_token_id = self.vqgan.quantize.n_e
            current_epoch = getattr(self.trainer, 'current_epoch', 0)
            base_mask_prob = min(0.25, 0.15 + 0.005 * current_epoch)
            masked_indices, mask = self.structured_masking(indices, base_mask_prob, patch_size=4)

            if self.lm_head is None:
                token_features, attn_outputs = self.gpt(masked_indices, return_features=True, return_attn=True)
                lm_logits = self.gpt.transformer.mlm_layer(
                    token_features, self.gpt.transformer.embed.word_embeddings.weight
                )
            else:
                token_features, attn_outputs = self.gpt(masked_indices, return_features=True, return_attn=True)
                lm_logits = self.lm_head(token_features)

            if mask.any():
                lm_loss = F.cross_entropy(lm_logits[mask], indices[mask])
                token_preds = torch.argmax(lm_logits[mask], dim=1)
                token_acc = (token_preds == indices[mask]).float().mean()
            else:
                lm_loss = torch.tensor(0.0, device=self.device)
                token_acc = torch.tensor(1.0, device=self.device)

            seg_outputs = []
            mask_pred_metrics = {}
            mask_pred_upsampled = None
            if self.use_gpt_mask_prediction:
                last_layer_attn = attn_outputs[-1] if attn_outputs else None
                mask_pred_logits = self.mask_prediction_head.get_spatial_logits(
                    token_features,
                    attention_maps=last_layer_attn,
                    height=H_q,
                    width=W_q
                )
                # mask_pred_logits = self.mask_prediction_head(token_features, last_layer_attn, H_q, W_q)
                token_mask_ce_loss = F.cross_entropy(mask_pred_logits, ds_masks)
                mask_pred_upsampled = F.interpolate(
                    mask_pred_logits, 
                    size=images.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                seg_outputs.append(mask_pred_upsampled)
                mask_pred_ce_loss = F.cross_entropy(mask_pred_upsampled, masks)
                mask_pred_dice_loss = self.dice_loss_fn(mask_pred_upsampled, masks)
                mask_pred_loss = token_mask_ce_loss + mask_pred_ce_loss + self.dice_loss_weight * mask_pred_dice_loss
                mask_preds = torch.argmax(mask_pred_upsampled, dim=1)
                mask_pred_dice = self.compute_dice(mask_preds, masks)
                self.mask_pred_accuracy(mask_preds, masks)
                self.mask_pred_iou(mask_preds, masks)
                mask_pred_metrics = {
                    'val/mask_pred_loss': mask_pred_loss,
                    'val/mask_pred_dice': mask_pred_dice,
                    'val/mask_pred_accuracy': self.mask_pred_accuracy,
                    'val/mask_pred_iou': self.mask_pred_iou,
                }

            attn_seg_metrics = {}
            attn_seg_logits = None
            if self.use_attn_supervision and attn_outputs:
                last_layer_attn = attn_outputs[-1]
                avg_attn = last_layer_attn.mean(dim=1)  # Average across heads
                
                # Process attention maps for segmentation
                B, T, _ = avg_attn.shape  # Get actual token count
                attn_features = avg_attn.view(B, T, T).permute(0, 2, 1)
                
                # Ensure module exists with correct dimensions
                self._ensure_attn_module(T)
                    
                attn_seg_token_logits = self.attn_segmentation_module(attn_features)
                
                attn_seg_token_logits = attn_seg_token_logits.permute(0, 2, 1).view(B, -1, H_q, W_q)
                token_attn_ce_loss = F.cross_entropy(attn_seg_token_logits, ds_masks)
                attn_seg_logits = F.interpolate(
                    attn_seg_token_logits,
                    size=images.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                seg_outputs.append(attn_seg_logits)
                attn_ce_loss = F.cross_entropy(attn_seg_logits, masks)
                attn_dice_loss = self.dice_loss_fn(attn_seg_logits, masks)
                attn_seg_loss = token_attn_ce_loss + attn_ce_loss + self.dice_loss_weight * attn_dice_loss
                attn_preds = torch.argmax(attn_seg_logits, dim=1)
                attn_dice = self.compute_dice(attn_preds, masks)
                attn_seg_metrics = {
                    'val/attn_seg_loss': attn_seg_loss,
                    'val/attn_seg_dice': attn_dice,
                }

            deepcut_logits = None
            if self.use_deepcut:
                S_all, deepcut_loss = self.deepcut_module(token_features, k=8)
                deepcut_logits = F.interpolate(S_all, size=images.shape[2:], mode='bilinear', align_corners=False)
                seg_outputs.append(deepcut_logits)
            else:
                deepcut_loss = torch.tensor(0.0, device=self.device)

            if self.use_codebook:
                seg_logits, _ = self.codebook_segmentation(token_features, indices.view(-1))
                seg_logits = seg_logits.view(B, self.hparams.segmentation_config['num_classes'], H_q, W_q)
                seg_logits = F.interpolate(seg_logits, size=images.shape[2:], mode='bilinear', align_corners=False)
                seg_outputs.append(seg_logits)

            if self.use_bipartite:
                B_assign, S_bip, bip_loss, bip_stats = self.diff_bipartite(token_features)
                bip_logits = B_assign.view(B, self.hparams.segmentation_config['num_classes'], H_q, W_q)
                bip_logits = F.interpolate(bip_logits, size=images.shape[2:], mode='bilinear', align_corners=False)
                seg_outputs.append(bip_logits)

            if len(seg_outputs) >= 2:
                if len(self.blend_params) != len(seg_outputs):
                    self.blend_params = nn.Parameter(torch.zeros(len(seg_outputs), device=self.device))
                blend_weights = F.softmax(self.blend_params, dim=0)
                final_seg_probs = sum(w * output for w, output in zip(blend_weights, seg_outputs))
            elif len(seg_outputs) == 1:
                final_seg_probs = seg_outputs[0]
                blend_weights = torch.tensor([1.0], device=self.device)
            else:
                final_seg_probs = None
                blend_weights = torch.tensor([], device=self.device)

            if final_seg_probs is not None:
                preds = torch.argmax(final_seg_probs, dim=1)
                loss_sup = F.cross_entropy(final_seg_probs, masks)
                val_dice = self.compute_dice(preds, masks)
                custom_ious = [compute_iou(preds[i], masks[i]) for i in range(preds.size(0))]
                custom_accs = [compute_accuracy(preds[i], masks[i]) for i in range(preds.size(0))]
                val_custom_iou = torch.stack(custom_ious).mean()
                val_custom_acc = torch.stack(custom_accs).mean()
                self.val_accuracy(preds, masks)
                self.val_iou(preds, masks)
            else:
                loss_sup = torch.tensor(0.0, device=self.device)
                val_dice = torch.tensor(0.0, device=self.device)
                val_custom_iou = torch.tensor(0.0, device=self.device)
                val_custom_acc = torch.tensor(0.0, device=self.device)
                preds = None

            recon_loss = F.mse_loss(recon, images_norm)

            log_dict = {
                'val/loss_sup': loss_sup,
                'val/loss_recon': recon_loss,
                'val/lm_loss': lm_loss,
                'val/lm_accuracy': token_acc,
                'val/vqgan/codebook_loss': codebook_loss,
                'val/seg/accuracy': self.val_accuracy,
                'val/seg/iou': self.val_iou,
                'val/dice': val_dice,
                'val/custom_iou': val_custom_iou,
                'val/custom_acc': val_custom_acc,
                'val/deepcut_loss': deepcut_loss if self.use_deepcut else 0.0,
            }
            log_dict.update(mask_pred_metrics)
            log_dict.update(attn_seg_metrics)
            for i, w in enumerate(blend_weights):
                log_dict[f'val/blend_weight_{i}'] = w.item()

            self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

            if batch_idx == 11 or batch_idx % 10 == 0:
                mask_preds_for_vis = torch.argmax(mask_pred_upsampled, dim=1) if mask_pred_upsampled is not None else None
                attn_preds_for_vis = torch.argmax(attn_seg_logits, dim=1) if attn_seg_logits is not None else None

                self._log_validation_images(
                    images,
                    recon,
                    preds,
                    masks,
                    masked_indices,
                    gpt_mask_preds=mask_preds_for_vis,
                    attn_mask_preds=attn_preds_for_vis
                )

            return loss_sup


    def _log_validation_images(self, images, reconstructions, pred_masks, masks, masked_indices=None,
                               gpt_mask_preds=None, attn_mask_preds=None):
        # Convert images from [-1, 1] to [0, 1] and clip to ensure valid range.
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, 0.0, 1.0)
        target_size = images.shape[2:]
        
        if masked_indices is not None:
            B, T = masked_indices.shape
            H_mask = W_mask = int(T ** 0.5)
            mask_vis = (masked_indices == self.mask_token_id).float().view(B, 1, H_mask, W_mask)
            mask_vis = F.interpolate(mask_vis, size=target_size, mode='nearest')
            mask_vis = mask_vis.repeat(1, 3, 1, 1)
        
        num_classes = self.hparams.segmentation_config['num_classes']
        cmap = plt.get_cmap('tab20', num_classes)
        
        vis_images = []
        num_vis = min(4, images.shape[0])
        for i in range(num_vis):
            vis_images.append(wandb.Image(images[i].cpu(), caption=f"Sample {i}"))
            vis_images.append(wandb.Image(reconstructions[i].cpu(), caption=f"Reconstruction {i}"))
            gt_mask_img = self._colorize_mask(masks[i].cpu(), cmap, num_classes)
            vis_images.append(wandb.Image(gt_mask_img, caption=f"GT Mask {i}"))
            if pred_masks is not None:
                pred_mask_img = self._colorize_mask(pred_masks[i].cpu(), cmap, num_classes)
                vis_images.append(wandb.Image(pred_mask_img, caption=f"Final Pred {i}"))
            if gpt_mask_preds is not None:
                gpt_mask_img = self._colorize_mask(gpt_mask_preds[i].cpu(), cmap, num_classes)
                vis_images.append(wandb.Image(gpt_mask_img, caption=f"GPT Mask Pred {i}"))
            if attn_mask_preds is not None:
                attn_mask_img = self._colorize_mask(attn_mask_preds[i].cpu(), cmap, num_classes)
                vis_images.append(wandb.Image(attn_mask_img, caption=f"Attn Mask Pred {i}"))
            if masked_indices is not None:
                vis_images.append(wandb.Image(mask_vis[i].cpu(), caption=f"Mask Pattern {i}"))
        
        self.logger.experiment.log({
            "val/visualizations": vis_images,
            "global_step": self.global_step,
            "epoch": self.trainer.current_epoch
        })

    def _create_mask_comparison(self, images, gt_masks, pred_masks=None, gpt_masks=None, attn_masks=None, num_vis=2):
        num_classes = self.hparams.segmentation_config['num_classes']
        cmap = plt.get_cmap('tab20', num_classes)
        comparison_images = []
        for i in range(min(num_vis, images.shape[0])):
            num_cols = 1 + sum(x is not None for x in [pred_masks, gpt_masks, attn_masks])
            fig, axs = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8))
            if num_cols == 1:
                axs = np.expand_dims(axs, axis=1)
            # Clip image values to [0, 1] before plotting.
            img_np = np.clip(images[i].cpu().permute(1, 2, 0).numpy(), 0, 1)
            axs[0, 0].imshow(img_np)
            axs[0, 0].set_title("Input Image")
            axs[0, 0].axis('off')
            
            gt_mask_np = gt_masks[i].cpu().numpy()
            gt_vis = self._colorize_mask(gt_mask_np, cmap, num_classes)
            axs[1, 0].imshow(gt_vis)
            axs[1, 0].set_title("Ground Truth")
            axs[1, 0].axis('off')
            
            col_idx = 1
            if pred_masks is not None:
                pred_mask_np = pred_masks[i].cpu().numpy()
                pred_vis = self._colorize_mask(pred_mask_np, cmap, num_classes)
                axs[0, col_idx].imshow(img_np)
                axs[0, col_idx].imshow(pred_vis, alpha=0.7)
                axs[0, col_idx].set_title("Final Prediction")
                axs[0, col_idx].axis('off')
                axs[1, col_idx].imshow(pred_vis)
                axs[1, col_idx].set_title(f"IoU: {compute_iou(pred_masks[i], gt_masks[i]):.3f}")
                axs[1, col_idx].axis('off')
                col_idx += 1
            if gpt_masks is not None:
                gpt_mask_np = gpt_masks[i].cpu().numpy()
                gpt_vis = self._colorize_mask(gpt_mask_np, cmap, num_classes)
                axs[0, col_idx].imshow(img_np)
                axs[0, col_idx].imshow(gpt_vis, alpha=0.7)
                axs[0, col_idx].set_title("GPT Mask Prediction")
                axs[0, col_idx].axis('off')
                axs[1, col_idx].imshow(gpt_vis)
                axs[1, col_idx].set_title(f"IoU: {compute_iou(gpt_masks[i], gt_masks[i]):.3f}")
                axs[1, col_idx].axis('off')
                col_idx += 1
            if attn_masks is not None:
                attn_mask_np = attn_masks[i].cpu().numpy()
                attn_vis = self._colorize_mask(attn_mask_np, cmap, num_classes)
                axs[0, col_idx].imshow(img_np)
                axs[0, col_idx].imshow(attn_vis, alpha=0.7)
                axs[0, col_idx].set_title("Attention Mask Prediction")
                axs[0, col_idx].axis('off')
                axs[1, col_idx].imshow(attn_vis)
                axs[1, col_idx].set_title(f"IoU: {compute_iou(attn_masks[i], gt_masks[i]):.3f}")
                axs[1, col_idx].axis('off')
            
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            comparison_images.append(wandb.Image(Image.open(buf), caption=f"Mask Comparison {i}"))
            plt.close(fig)
        return comparison_images

    def _colorize_mask(self, mask, cmap, num_classes):
        """Convert a class-index mask to a colorized RGB image (values in [0,1])."""
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        h, w = mask_np.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(num_classes):
            mask_c = mask_np == c
            if mask_c.any():
                rgb_c = np.array(cmap(c)[:3])  # Already in [0,1]
                for j in range(3):
                    rgb_mask[mask_c, j] = rgb_c[j]
        return np.clip(rgb_mask, 0, 1)

    def _log_generated_images(self, orig_images, gen_images):
        orig_images = torch.clamp((orig_images + 1.0) / 2.0, 0.0, 1.0)
        gen_images = torch.clamp((gen_images + 1.0) / 2.0, 0.0, 1.0)
        vis_images = [wandb.Image(orig_images[i].cpu(), caption=f"Original {i}") for i in range(min(2, orig_images.shape[0]))]
        vis_images += [wandb.Image(gen_images[i].cpu(), caption=f"Generated {i}") for i in range(min(2, gen_images.shape[0]))]
        self.logger.experiment.log({
            "generation/samples": vis_images,
            "global_step": self.global_step,
            "epoch": self.trainer.current_epoch
        })


    # -----------------------------
    # Autoregressive Sampling Method
    # -----------------------------
    def sample_autoregressive_tokens(self, c_indices, z_indices_shape, temperature=1.0, top_k=100,
                                     window_size=16, stride=8, return_intermediates=False):
        if hasattr(self.gpt, 'generate') and not return_intermediates:
            device = c_indices.device
            batch_size = c_indices.shape[0]
            H, W = z_indices_shape[1], z_indices_shape[2]
            flat_c_indices = c_indices if c_indices.dim() == 2 else c_indices.reshape(batch_size, -1)
            max_new_tokens = H * W
            generated = self.gpt.generate(
                context=flat_c_indices,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            z_indices = generated[:, -max_new_tokens:].reshape(batch_size, H, W)
            return z_indices, None
        else:
            device = c_indices.device
            batch_size = c_indices.shape[0]
            H, W = z_indices_shape[1], z_indices_shape[2]
            z_indices = torch.randint(0, self.vqgan.quantize.n_e, (batch_size, H, W), device=device)
            if c_indices.dim() == 2:
                c_T = c_indices.shape[1]
                c_H = c_W = int(c_T ** 0.5)
                c_indices = c_indices.reshape(batch_size, c_H, c_W)
            intermediates = [] if return_intermediates else None
            for i in range(0, H):
                local_i = i if i <= window_size // 2 else (window_size - (H - i) if H - i < window_size // 2 else window_size // 2)
                i_start = max(0, i - local_i)
                i_end = min(H, i_start + window_size)
                for j in range(0, W):
                    local_j = j if j <= window_size // 2 else (window_size - (W - j) if W - j < window_size // 2 else window_size // 2)
                    j_start = max(0, j - local_j)
                    j_end = min(W, j_start + window_size)
                    c_patch = c_indices[:, i_start:i_end, j_start:j_end]
                    z_patch = z_indices[:, i_start:i_end, j_start:j_end]
                    c_patch_flat = c_patch.reshape(batch_size, -1)
                    z_patch_flat = z_patch.reshape(batch_size, -1)
                    patch = torch.cat((c_patch_flat, z_patch_flat), dim=1)
                    if self.lm_head is None:
                        logits, _ = self.gpt.transformer(patch[:, :-1])
                    else:
                        features = self.gpt.transformer.embed(patch[:, :-1])
                        for layer in self.gpt.transformer.layers:
                            features = layer(features)
                        logits = self.lm_head(features)
                    patch_area = (j_end - j_start) * (i_end - i_start)
                    local_pos = local_i * (j_end - j_start) + local_j
                    logits = logits[:, -patch_area:, :].reshape(batch_size, i_end - i_start, j_end - j_start, -1)
                    current_token_logits = logits[:, local_i, local_j, :]
                    current_token_logits = current_token_logits / temperature
                    if top_k is not None:
                        current_token_logits = self.top_k_logits(current_token_logits, top_k)
                    probs = F.softmax(current_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    z_indices[:, i, j] = next_token
                    if return_intermediates and (i * W + j) % stride == 0:
                        z_code_shape = (batch_size, self.vqgan.embed_dim, H, W)
                        x_sample = self.decode_to_img(z_indices.reshape(batch_size, -1), z_code_shape)
                        intermediates.append(x_sample)
            return z_indices, intermediates

    def top_k_logits(self, logits, k):
        if hasattr(self.gpt, 'top_k_logits'):
            return self.gpt.top_k_logits(logits, k)
        else:
            v, ix = torch.topk(logits, k)
            out = logits.clone()
            out[out < v[..., [-1]]] = -float('Inf')
            return out

    # -----------------------------
    # Decoding method
    # -----------------------------
    def decode_to_img(self, z_indices, z_shape):
        B, C, H, W = z_shape
        if z_indices.dim() == 1:
            z_indices = z_indices.reshape(B, H * W)
        z_indices = z_indices.clamp(0, self.vqgan.quantize.n_e - 1)
        z_one_hot = F.one_hot(z_indices, num_classes=self.vqgan.quantize.n_e).float()
        z_one_hot = z_one_hot @ self.vqgan.quantize.embedding.weight
        z = z_one_hot.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        img = self.vqgan.decode(z)
        return img

    # -----------------------------
    # Epoch hooks
    # -----------------------------
    def on_epoch_start(self):
        self.mask_token_id = self.vqgan.quantize.n_e
        current_epoch = self.trainer.current_epoch
        opt = self.optimizers()[0]
        if current_epoch > 0 and current_epoch % 10 == 0:
            for param_group in opt.param_groups:
                if 'vqgan' in param_group.get('name', '') and current_epoch > 30:
                    param_group['lr'] *= 0.8
                elif 'gpt' in param_group.get('name', '') and current_epoch < 20:
                    param_group['lr'] = min(param_group['lr'] * 1.1, self.learning_rate * 2)
        if current_epoch > 50:
            self.gradient_accumulation_steps = min(4, self.gradient_accumulation_steps + 1)

    def on_validation_epoch_end(self):
        acc = self.val_accuracy.compute()
        iou = self.val_iou.compute()
        self.log("val/seg/accuracy", acc, prog_bar=True)
        self.log("val/seg/iou", iou, prog_bar=True)
        self.val_accuracy.reset()
        self.val_iou.reset()
        if self.trainer.current_epoch >= 10:
            if not hasattr(self, 'best_iou') or iou > self.best_iou:
                self.best_iou = iou
                self.trainer.save_checkpoint(f"best_iou_model_epoch{self.trainer.current_epoch}.ckpt")
                self.log("val/best_iou", iou)
                self.log("val/best_iou_epoch", self.trainer.current_epoch)

    def freeze_vqgan(self):
        for param in self.vqgan.parameters():
            param.requires_grad = False

        opt = self.optimizers()[0]
        for param_group in opt.param_groups:
            if 'vqgan' in param_group.get('name', '') or 'discriminator' in param_group.get('name', ''):
                param_group['lr'] = 0

    def unfreeze_vqgan(self):
        for param in self.vqgan.parameters():
            param.requires_grad = True

        opt = self.optimizers()[0]
        for param_group in opt.param_groups:
            if 'vqgan' in param_group.get('name', ''):
                param_group['lr'] = self.learning_rate * 0.01
            elif 'discriminator' in param_group.get('name', ''):
                param_group['lr'] = self.learning_rate * 0.01

    # -----------------------------
    # Optimizer configuration
    # -----------------------------
    def configure_optimizers(self):
        if self.lm_head is None:
            gpt_params = list(self.gpt.parameters())
        else:
            gpt_params = list(self.gpt.parameters()) + list(self.lm_head.parameters())
        
        # Collect mask prediction and attention segmentation parameters
        mask_pred_params = []
        if self.use_gpt_mask_prediction and hasattr(self, 'mask_prediction_head'):
            mask_pred_params.extend(list(self.mask_prediction_head.parameters()))
        
        attn_seg_params = []
        if self.use_attn_supervision:
            # The module might not exist yet, so we'll create a dummy one just for initialization
            if self.attn_segmentation_module is None:
                # Create a temporary module with expected dimensions for optimizer initialization
                temp_module = self.build_attn_seg_module(self.expected_token_count).to(self.device)
                attn_seg_params.extend(list(temp_module.parameters()))
            else:
                attn_seg_params.extend(list(self.attn_segmentation_module.parameters()))
            
        param_groups = [
            {
                'params': list(self.vqgan.encoder.parameters()) +
                        list(self.vqgan.decoder.parameters()) +
                        list(self.vqgan.quantize.parameters()) +
                        list(self.vqgan.quant_conv.parameters()) +
                        list(self.vqgan.post_quant_conv.parameters()),
                'lr': self.learning_rate * 0.01,
                'betas': (0.5, 0.9),
                'name': 'vqgan_main'
            },
            {
                'params': self.vqgan.discriminator.parameters(),
                'lr': self.learning_rate * 0.01,
                'betas': (0.5, 0.9),
                'name': 'discriminator'
            },
            {
                'params': gpt_params +
                        (list(self.codebook_segmentation.parameters()) if self.use_codebook else []) +
                        (list(self.deepcut_module.parameters()) if self.use_deepcut else []) +
                        (list(self.diff_bipartite.parameters()) if self.use_bipartite else []) +
                        mask_pred_params +
                        attn_seg_params +
                        [self.blend_params, self.blend_alpha],
                'lr': self.learning_rate,
                'betas': (0.9, 0.95),
                'weight_decay': 0.1,
                'name': 'gpt_seg'
            }
        ]
        
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.learning_rate * 0.01
        )
        scheduler_config = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        return [optimizer], [scheduler_config]


    # We'll also need to update how we handle optimizer during training
    def on_before_optimizer_step(self, optimizer):
        """Called before `optimizer.step()`. 
        Used to update optimizer with dynamically created module parameters."""
        
        # Check if modules were created after optimizer initialization
        if self.use_attn_supervision and self.attn_segmentation_module is not None:
            # Get the existing parameter group
            gpt_seg_group = None
            for group in optimizer.param_groups:
                if group.get('name') == 'gpt_seg':
                    gpt_seg_group = group
                    break
                    
            if gpt_seg_group is not None:
                # Check if attn parameters are already in optimizer
                param_ids = {id(p) for p in gpt_seg_group['params']}
                
                # Add new parameters if they don't exist
                for param in self.attn_segmentation_module.parameters():
                    if id(param) not in param_ids:
                        gpt_seg_group['params'].append(param)