import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from contextlib import contextmanager
import wandb
from src.vqa.vqagan import VQGAN


class VQGANPretraining(pl.LightningModule):
    def __init__(self, vqgan_config: dict, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.vqgan = VQGAN(**vqgan_config)
        self.learning_rate = learning_rate

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

    def training_step(self, batch, batch_idx):
        images, _ = batch
        images = images.to(self.device)
        images_norm = 2.0 * images - 1.0  # Normalize images to [-1,1]
        opt_vq, opt_disc = self.optimizers()
        with self.ema_scope():
            # Generator/encoder-decoder update
            opt_vq.zero_grad()
            loss_vq, vq_log_dict = self.vqgan(images_norm, optimizer_idx=0, global_step=self.global_step)
            self.manual_backward(loss_vq)
            # Clip gradients for all generator parameters
            torch.nn.utils.clip_grad_norm_(list(self.vqgan.encoder.parameters()) +
                                             list(self.vqgan.decoder.parameters()) +
                                             list(self.vqgan.quantize.parameters()) +
                                             list(self.vqgan.quant_conv.parameters()) +
                                             list(self.vqgan.post_quant_conv.parameters()), 1.0)
            opt_vq.step()

            # Discriminator update (after disc_start steps)
            loss_disc = None
            if self.global_step >= self.vqgan.loss.disc_start:
                opt_disc.zero_grad()
                steps_since_disc_start = self.global_step - self.vqgan.loss.disc_start
                increments = steps_since_disc_start // 1000
                disc_weight = min(1.0, 0.01 * (increments + 1))
                loss_disc, disc_log_dict = self.vqgan(images_norm, optimizer_idx=1, global_step=self.global_step)
                if loss_disc is not None:
                    loss_disc = disc_weight * loss_disc
                    self.manual_backward(loss_disc)
                    torch.nn.utils.clip_grad_norm_(self.vqgan.discriminator.parameters(), 1.0)
                    opt_disc.step()
        log_dict = {
            'train/loss_vq': loss_vq,
            'train/loss_disc': loss_disc if loss_disc is not None else 0.0,
        }
        if vq_log_dict is not None:
            for k, v in vq_log_dict.items():
                log_dict[f'train/vq/{k}'] = v
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True)
        return loss_vq

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        images = images.to(self.device)
        images_norm = 2.0 * images - 1.0
        with torch.no_grad(), self.ema_scope():
            recon, codebook_loss, info = self.vqgan(images_norm)
            loss_vq, vq_log_dict = self.vqgan(images_norm, optimizer_idx=0, global_step=self.global_step)
            recon_loss = F.mse_loss(recon, images_norm)
            self.log('val/loss_recon', recon_loss, prog_bar=True, sync_dist=True)
            self.log('val/loss_vq', loss_vq, prog_bar=True, sync_dist=True)
            self.log('val/codebook_loss', codebook_loss, prog_bar=True, sync_dist=True)
            if vq_log_dict is not None:
                for k, v in vq_log_dict.items():
                    self.log(f'val/vq/{k}', v, prog_bar=False, sync_dist=True)
            if batch_idx == 0:
                images_disp = (images + 1.0) / 2.0
                reconstructions = (recon + 1.0) / 2.0
                vis_images = []
                num_vis = min(4, images.shape[0])
                for i in range(num_vis):
                    vis_images.append(wandb.Image(images_disp[i].cpu(), caption=f"Sample {i}"))
                    vis_images.append(wandb.Image(reconstructions[i].cpu(), caption=f"Reconstruction {i}"))
                self.logger.experiment.log({
                    "val/visualizations": vis_images,
                    "global_step": self.global_step,
                    "epoch": self.trainer.current_epoch
                })
            return recon_loss

    def configure_optimizers(self):
        optimizer_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.quantize.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=self.learning_rate,
            betas=(0.5, 0.9)
        )
        optimizer_disc = torch.optim.Adam(
            self.vqgan.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.9)
        )
        scheduler_vq = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_vq, 
            T_max=self.trainer.max_epochs, 
            eta_min=self.learning_rate * 0.001
        )
        return [optimizer_vq, optimizer_disc], [{
            "scheduler": scheduler_vq,
            "interval": "epoch",
            "frequency": 1
        }]