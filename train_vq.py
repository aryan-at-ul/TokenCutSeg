import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.vq_model_trainer import VQGANPretraining
from src.utils.data_loading import create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--train-data-dir', type=str, default='/media/annatar/OLDHDD/images_skin_cancer_data_gan/train') #HAM10000 dataset 10k images
    parser.add_argument('--train-mask-dir', type=str, default='/media/annatar/OLDHDD/images_skin_cancer_data_gan/train')
    parser.add_argument('--val-data-dir', type=str, default='/media/annatar/OLDHDD/images_skin_cancer_data_gan/val')
    parser.add_argument('--val-mask-dir', type=str, default='/media/annatar/OLDHDD/images_skin_cancer_data_gan/val')
    parser.add_argument('--image-size', type=int, default=224)

    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=int, default=32)
    
    # Model arguments - VQ-GAN
    parser.add_argument('--vq-embed-dim', type=int, default=256) # final model uses 256
    parser.add_argument('--vq-n-embed', type=int, default=64) # this is the codebook size, 32x32 = 1024
    parser.add_argument('--vq-hidden-channels', type=int, default=128)
    parser.add_argument('--vq-n-res-blocks', type=int, default=2)
    parser.add_argument('--disc-start', type=int, default=10000)
    parser.add_argument('--disc-weight', type=float, default=0.5)
    parser.add_argument('--perceptual-weight', type=float, default=0.8)
    parser.add_argument('--codebook-weight', type=float, default=1.0)
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='vqgan_pretrain_64')
    parser.add_argument('--experiment-name', type=str, default='vqgan_pretrain_isic_64')
    
    return parser.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(42)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure VQ-GAN
    vqgan_config = {
        'n_embed': args.vq_n_embed,
        'embed_dim': args.vq_embed_dim,
        'hidden_channels': args.vq_hidden_channels,
        'n_res_blocks': args.vq_n_res_blocks,
        'disc_start': args.disc_start,
        'disc_weight': args.disc_weight,
        'perceptual_weight': args.perceptual_weight,
        'codebook_weight': args.codebook_weight
    }
    
    model = VQGANPretraining(
        vqgan_config=vqgan_config,
        learning_rate=args.learning_rate
    )
    
    train_loader, val_loader = create_dataloaders(
        train_data_dir=args.train_data_dir,
        train_mask_dir=args.train_mask_dir,
        val_data_dir=args.val_data_dir,
        val_mask_dir=args.val_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        vqtrain=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename=f'{args.experiment_name}' + '-{epoch:02d}-{val_loss_vq:.4f}',
        monitor='val/loss_vq',
        mode='min',
        save_top_k=2,
        save_last=True,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    early_stopping = EarlyStopping(
        monitor='val/loss_vq',
        patience=10,
        mode='min',
        verbose=True
    )
    
    # Configure logger
    wandb_logger = WandbLogger(
        project=args.experiment_name,
        name=f"vqgan_pretrain-{args.image_size}-bs{args.batch_size}-lr{args.learning_rate}",
        config=args
    )
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10,
        strategy='auto',
        detect_anomaly=True,
        num_sanity_val_steps=1
    )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()