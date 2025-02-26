import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.vq_seg_model import VQGPTSegmentation
from src.utils.data_loading import create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-data-dir', type=str, default='/media/annatar/OLDHDD/isic2018/train')
    parser.add_argument('--train-mask-dir', type=str, default='/media/annatar/OLDHDD/isic2018/train_masks')
    parser.add_argument('--val-data-dir', type=str, default='/media/annatar/OLDHDD/isic2018/val')
    parser.add_argument('--val-mask-dir', type=str, default='/media/annatar/OLDHDD/isic2018/val_masks')
    parser.add_argument('--image-size', type=int, default=224)
    
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    

    parser.add_argument('--vqgan-checkpoint', type=str, default='vqgan_pretrain_512/checkpoints/vqgan_pretrain_isic_512-epoch=20-val_loss_vq=0.0000.ckpt',
                        help='Path to pretrained VQGAN checkpoint')
    parser.add_argument('--vq-embed-dim', type=int, default=256)
    parser.add_argument('--vq-n-embed', type=int, default=512)
    parser.add_argument('--vq-hidden-channels', type=int, default=128)
    parser.add_argument('--vq-n-res-blocks', type=int, default=2)
    

    parser.add_argument('--gpt-n-layer', type=int, default=12) # 12
    parser.add_argument('--gpt-n-head', type=int, default=4) # 4
    parser.add_argument('--gpt-n-embd', type=int, default=256)
    

    parser.add_argument('--use-deepcut', type=bool, default=True)
    parser.add_argument('--deepcut-loss-weight', type=float, default=0.2)
    
    parser.add_argument('--output-dir', type=str, default='segmentation_training_2018_codebook512')
    parser.add_argument('--experiment-name', type=str, default='segmentation_isic2018_codebook512')
    
    return parser.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure VQ-GAN
    vqgan_config = {
        'n_embed': args.vq_n_embed,
        'embed_dim': args.vq_embed_dim,
        'hidden_channels': args.vq_hidden_channels,
        'n_res_blocks': args.vq_n_res_blocks,
    }
    
    # Configure ImageGPT
    gpt_config = {
        'vocab_size': args.vq_n_embed + 1,  # +1 for mask token
        'block_size': (args.image_size // 4) ** 2,
        'n_layer': args.gpt_n_layer,
        'n_head': args.gpt_n_head,
        'n_embd': args.gpt_n_embd
    }
    
    # Configure Segmentation
    segmentation_config = {
        'num_classes': 2,  # Binary segmentation for ISIC
        'use_deepcut': args.use_deepcut,
        'deepcut_loss_weight': args.deepcut_loss_weight,
    }
    
    # Create model
    model = VQGPTSegmentation(
        vqgan_path=args.vqgan_checkpoint,
        vqgan_config=vqgan_config,
        gpt_config=gpt_config,
        segmentation_config=segmentation_config,
        learning_rate=args.learning_rate
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_data_dir=args.train_data_dir,
        train_mask_dir=args.train_mask_dir,
        val_data_dir=args.val_data_dir,
        val_mask_dir=args.val_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename=f'{args.experiment_name}' + '-{epoch:02d}-{val_iou:.4f}',
        monitor='val/iou',
        mode='max',
        save_top_k=2,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Configure logger
    wandb_logger = WandbLogger(
        project=args.experiment_name,
        name=f"seg_training-{args.image_size}-bs{args.batch_size}-lr{args.learning_rate}"
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10,
        strategy='auto',
        num_sanity_val_steps=1
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()