# # src/vqa/perceptual.py

# import torch
# import torch.nn as nn
# from torchvision import models
# import torch.nn.functional as F

# class VGGPerceptualLoss(nn.Module):
#     def __init__(self, resize=False):
#         super().__init__()
#         # Use VGG16 loaded from pretrained weights
#         vgg = models.vgg16(pretrained=True)
#         blocks = []
#         blocks.append(vgg.features[:4].eval())
#         blocks.append(vgg.features[4:9].eval())
#         blocks.append(vgg.features[9:16].eval())
#         blocks.append(vgg.features[16:23].eval())
#         blocks.append(vgg.features[23:30].eval())
        
#         for bl in blocks:
#             for p in bl.parameters():
#                 p.requires_grad = False
                
#         self.blocks = nn.ModuleList(blocks)
#         self.resize = resize
#         self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

#     def preprocess(self, x):
#         # Expect input in range [-1, 1]
#         x = (x + 1) / 2  # Convert to [0, 1]
#         x = (x - self.mean) / self.std
#         return x

#     def forward(self, input, target, normalize=True):
#         if normalize:
#             input = self.preprocess(input)
#             target = self.preprocess(target)
            
#         if self.resize:
#             input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
#             target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
        
#         loss = 0.0
#         x = input
#         y = target
        
#         for block in self.blocks:
#             x = block(x)
#             y = block(y)
#             loss += F.l1_loss(x, y)
            
#         return loss

# class VQLPIPSWithDiscriminator(nn.Module):
#     def __init__(self, disc_start, codebook_weight=1.0, pixel_weight=1.0,
#                  perceptual_weight=1.0, disc_weight=1.0):
#         super().__init__()
#         self.codebook_weight = codebook_weight
#         self.pixel_weight = pixel_weight
#         self.perceptual_weight = perceptual_weight
#         self.disc_weight = disc_weight
#         self.disc_start = disc_start

#         self.perceptual_loss = VGGPerceptualLoss()

#     def forward(self, codebook_loss, inputs, reconstructions, g_loss, d_loss, 
#                 optimizer_idx, global_step, last_layer=None):
#         # Reconstruction loss
#         rec_loss = torch.abs(inputs - reconstructions).mean()  # Aggregate to scalar
#         if self.perceptual_weight > 0:
#             p_loss = self.perceptual_loss(inputs, reconstructions)
#             rec_loss = rec_loss + self.perceptual_weight * p_loss  # Both are scalars

#         # Generator loss
#         if optimizer_idx == 0:
#             # After disc_start steps, add GAN loss
#             if global_step >= self.disc_start:
#                 loss = (rec_loss + 
#                        self.codebook_weight * codebook_loss + 
#                        self.disc_weight * g_loss)
#             else:
#                 loss = rec_loss + self.codebook_weight * codebook_loss
#                 g_loss = torch.tensor(0.0, device=rec_loss.device)

#             log = {
#                 "total_loss": loss,             # Scalar
#                 "rec_loss": rec_loss,           # Scalar
#                 "g_loss": g_loss,               # Scalar
#                 "codebook_loss": codebook_loss, # Scalar
#             }
#             return loss, log

#         # Discriminator loss
#         if optimizer_idx == 1:
#             if global_step >= self.disc_start:
#                 log = {
#                     "d_loss": d_loss,  # Scalar
#                 }
#                 return d_loss, log
#             else:
#                 # No discriminator training before disc_start steps
#                 d_loss = torch.tensor(0.0, device=rec_loss.device)
#                 log = {
#                     "d_loss": d_loss,  # Scalar
#                 }
#                 return d_loss, log




#second imp blue schmee type 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        blocks.append(vgg.features[23:30].eval())
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.blocks = nn.ModuleList(blocks)
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, x):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x[:, :3, ...]
        x = torch.clamp((x + 1) / 2, 0, 1)  # Convert to [0, 1] with clipping
        x = (x - self.mean) / (self.std + 1e-6)
        return x

    def forward(self, input, target, normalize=True):
        if normalize:
            input = self.preprocess(input)
            target = self.preprocess(target)
            
        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        loss = 0.0
        x = input
        y = target
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            layer_loss = F.l1_loss(x, y)
            # Clip individual layer losses
            loss += torch.clamp(layer_loss, 0, 10)
            
        return loss

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixel_weight=1.0,
                 perceptual_weight=1.0, disc_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight
        self.disc_start = disc_start

        self.perceptual_loss = VGGPerceptualLoss()
        self.style_weight = 0.1
        self.eps = 1e-6

    def _gram_matrix(self, x):
        if not torch.isfinite(x).all():
            return torch.zeros_like(x)
            
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w + self.eps)
        return torch.clamp(gram, -1e3, 1e3)

    def _compute_edge_weights(self, x):
        if not torch.isfinite(x).all():
            return torch.ones_like(x[:, :1, :, :])
            
        kernel_x = torch.tensor([[-0.25, 0., 0.25]], device=x.device).view(1, 1, 1, 3)
        kernel_y = torch.tensor([[-0.25], [0.], [0.25]], device=x.device).view(1, 1, 3, 1)
        
        pad_x = F.pad(x, (1, 1, 0, 0), mode='reflect')
        pad_y = F.pad(x, (0, 0, 1, 1), mode='reflect')
        
        n_channels = x.size(1)
        try:
            grad_x = F.conv2d(pad_x, kernel_x.repeat(n_channels, 1, 1, 1), groups=n_channels)
            grad_y = F.conv2d(pad_y, kernel_y.repeat(n_channels, 1, 1, 1), groups=n_channels)
            
            grad_x = torch.clamp(grad_x, -1, 1)
            grad_y = torch.clamp(grad_y, -1, 1)
            
            grad_mag = torch.clamp(
                torch.abs(grad_x).mean(1, keepdim=True) + 
                torch.abs(grad_y).mean(1, keepdim=True),
                0, 1
            )
            weights = torch.exp(-grad_mag * 5)
            return weights.clamp(0.2, 1.0)
        except RuntimeError:
            return torch.ones_like(x[:, :1, :, :])

    def forward(self, codebook_loss, inputs, reconstructions, g_loss, d_loss, 
                optimizer_idx, global_step, last_layer=None, disc_factor=None):
        try:
            # Compute reconstruction loss
            rec_loss = torch.abs(inputs - reconstructions)
            rec_loss = torch.clamp(rec_loss, 0, 1)
            
            # Add edge awareness
            edge_weight = self._compute_edge_weights(inputs)
            rec_loss = (rec_loss * (1 + edge_weight)).mean()
            
            # Add perceptual loss if configured
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                p_loss = torch.clamp(p_loss, 0, 10)
                rec_loss = rec_loss + self.perceptual_weight * p_loss

            # Generator path
            if optimizer_idx == 0:
                # Always include reconstruction and codebook loss
                loss = rec_loss + self.codebook_weight * codebook_loss
                
                # After disc_start steps, add GAN loss
                if global_step >= self.disc_start:
                    g_loss = torch.clamp(g_loss, -1, 1)
                    loss = loss + self.disc_weight * g_loss
                else:
                    # Create zero tensor with gradients
                    g_loss = torch.zeros_like(rec_loss, requires_grad=True)

                log = {
                    "total_loss": loss.detach(),
                    "rec_loss": rec_loss.detach(),
                    "g_loss": g_loss.detach(),
                    "codebook_loss": codebook_loss.detach()
                }
                return loss, log

            # Discriminator path
            if optimizer_idx == 1:
                if global_step >= self.disc_start:
                    d_loss = torch.clamp(d_loss, -1, 1)
                    log = {
                        "d_loss": d_loss.detach()
                    }
                    return d_loss, log
                else:
                    # Create zero tensor with gradients
                    d_loss = torch.zeros_like(rec_loss, requires_grad=True)
                    log = {
                        "d_loss": d_loss.detach()
                    }
                    return d_loss, log

        except Exception as e:
            print(f"Unexpected error in loss computation: {e}")
            # Return zero tensor with gradients
            return torch.zeros_like(rec_loss, requires_grad=True), {"total_loss": 0.0}



