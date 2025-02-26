import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.ablation_trainer import VQSegmentationModel  
from src.utils.data_loading import create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser()
    

    parser.add_argument('--train-data-dir', type=str, default='/media/annatar/OLDHDD/isic2016/train')
    parser.add_argument('--train-mask-dir', type=str, default='/media/annatar/OLDHDD/isic2016/train_masks')
    parser.add_argument('--val-data-dir', type=str, default='/media/annatar/OLDHDD/isic2016/val')
    parser.add_argument('--val-mask-dir', type=str, default='/media/annatar/OLDHDD/isic2016/val_masks')
    parser.add_argument('--image-size', type=int, default=224)
    

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    
    # Model arguments - VQ-GAN
    parser.add_argument('--vqgan-checkpoint', type=str, default='vqgan_model_pretrained.ckpt',
                        help='Path to pretrained VQGAN checkpoint')
    parser.add_argument('--vq-embed-dim', type=int, default=256)
    parser.add_argument('--vq-n-embed', type=int, default=1024)
    parser.add_argument('--vq-hidden-channels', type=int, default=128)
    parser.add_argument('--vq-n-res-blocks', type=int, default=2)
    
    # Model arguments - ImageGPT
    parser.add_argument('--gpt-n-layer', type=int, default=12)
    parser.add_argument('--gpt-n-head', type=int, default=4)
    parser.add_argument('--gpt-n-embd', type=int, default=256)
    
    # Model arguments - Segmentation
    parser.add_argument('--use-deepcut', type=bool, default=False)  # Changed default to False
    parser.add_argument('--deepcut-loss-weight', type=float, default=0.2)
    
    # Ablation study arguments - NEW
    parser.add_argument('--ablation-mode', type=str, default='vqgan-gpt-mlm',
                       choices=['vqgan-only', 'vqgan-gpt', 'vqgan-gpt-mlm'],
                       help='Which model configuration to use')
    parser.add_argument('--mlm-weight', type=float, default=1.0,
                       help='Weight for masked language modeling loss')
    parser.add_argument('--mask-prob', type=float, default=0.15,
                       help='Probability of masking tokens for MLM')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='vqgan_gpt_mlm_segmentation_training_2017_ablation')
    parser.add_argument('--experiment-name', type=str, default='vqgan_gpt_mlm_segmentation_isic2017_ablation')
    
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
        'n_embd': args.gpt_n_embd,
        # 'mask_prob': args.mask_prob,
        # 'mask_token_id': 0,  # Using 0 as the mask token ID
        # 'loss_weight': args.mlm_weight
    }
    
    # Configure Segmentation
    segmentation_config = {
        'num_classes': 2,  # Binary segmentation for ISIC
        'use_deepcut': args.use_deepcut,
        'deepcut_loss_weight': args.deepcut_loss_weight,
    }
    
    # Set up ablation study flags
    use_gpt = args.ablation_mode != 'vqgan-only'
    use_mlm = args.ablation_mode == 'vqgan-gpt-mlm'
    
    # Create model
    model = VQSegmentationModel(
        vqgan_path=args.vqgan_checkpoint,
        vqgan_config=vqgan_config,
        gpt_config=gpt_config,
        segmentation_config=segmentation_config,
        learning_rate=args.learning_rate,
        use_gpt=use_gpt,
        use_mlm=use_mlm
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
        filename=f'{args.experiment_name}-{args.ablation_mode}' + '-{epoch:02d}-{val_iou:.4f}',
        monitor='val/iou',
        mode='max',
        save_top_k=2,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Configure logger
    wandb_logger = WandbLogger(
        project=args.experiment_name,
        name=f"{args.ablation_mode}-{args.image_size}-bs{args.batch_size}-lr{args.learning_rate}"
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
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