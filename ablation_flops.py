import os
import argparse
import torch
import pytorch_lightning as pl
from thop import profile, clever_format

from src.ablation_trainer import VQSegmentationModel


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--train-data-dir', type=str, default='/media/annatar/OLDHDD/isic2016/train')
    parser.add_argument('--train-mask-dir', type=str, default='/media/annatar/OLDHDD/isic2016/train_masks')
    parser.add_argument('--val-data-dir', type=str, default='/media/annatar/OLDHDD/isic2016/test')
    parser.add_argument('--val-mask-dir', type=str, default='/media/annatar/OLDHDD/isic2016/test_masks')
    parser.add_argument('--image-size', type=int, default=224)
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    
    # Model arguments - VQ-GAN
    parser.add_argument('--vqgan-checkpoint', type=str, default='vqgan_pretrain_64/checkpoints/vqgan_pretrain_isic_64-epoch=20-val_loss_vq=0.0000.ckpt', #full >> vqgan_model_pretrained.ckpt 1024
                        help='Path to pretrained VQGAN checkpoint')
    parser.add_argument('--vq-embed-dim', type=int, default=256)
    parser.add_argument('--vq-n-embed', type=int, default=64) #1024
    parser.add_argument('--vq-hidden-channels', type=int, default=128)
    parser.add_argument('--vq-n-res-blocks', type=int, default=2)
    
    # Model arguments - ImageGPT
    parser.add_argument('--gpt-n-layer', type=int, default=12)
    parser.add_argument('--gpt-n-head', type=int, default=4)
    parser.add_argument('--gpt-n-embd', type=int, default=256)
    
    # Model arguments - Segmentation
    parser.add_argument('--use-deepcut', type=bool, default=False)
    parser.add_argument('--deepcut-loss-weight', type=float, default=0.2)
    
    # Ablation study arguments
    parser.add_argument('--ablation-mode', type=str, default='vqgan-only',
                       choices=['vqgan-only', 'vqgan-gpt', 'vqgan-gpt-mlm'],
                       help='Which model configuration to use')
    parser.add_argument('--mlm-weight', type=float, default=1.0,
                       help='Weight for masked language modeling loss')
    parser.add_argument('--mask-prob', type=float, default=0.15,
                       help='Probability of masking tokens for MLM')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='model_profiling_results')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                       help='Path to a specific model checkpoint to profile')
    
    return parser.parse_args()


def print_model_info(model, args, device):
    """
    Print detailed information about model parameters and FLOPs
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n==== MODEL STATISTICS ====")
    print(f"Ablation mode: {args.ablation_mode}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    try:
        # Profile VQGAN
        dummy_img = torch.randn(1, 3, args.image_size, args.image_size).to(device)
        macs_vq, params_vq = profile(model.vqgan, inputs=(dummy_img,), verbose=False)
        macs_vq, params_vq = clever_format([macs_vq, params_vq], "%.3f")
        print("\nVQGAN:")
        print(f"  FLOPs: {macs_vq}")
        print(f"  Parameters: {params_vq}")
        
        # Profile ImageGPT if used
        if args.ablation_mode != 'vqgan-only' and hasattr(model, 'gpt'):
            dummy_indices = torch.randint(0, args.vq_n_embed, (1, (args.image_size // 4) ** 2)).to(device)
            macs_gpt, params_gpt = profile(model.gpt, inputs=(dummy_indices,), verbose=False)
            macs_gpt, params_gpt = clever_format([macs_gpt, params_gpt], "%.3f")
            print("\nImageGPT:")
            print(f"  FLOPs: {macs_gpt}")
            print(f"  Parameters: {params_gpt}")
        
        # Profile DeepCut if used
        if args.use_deepcut and hasattr(model, 'deepcut_module'):
            dummy_features = torch.randn(1, (args.image_size // 4) ** 2, args.gpt_n_embd).to(device)
            macs_deepcut, params_deepcut = profile(model.deepcut_module, inputs=(dummy_features,), verbose=False)
            macs_deepcut, params_deepcut = clever_format([macs_deepcut, params_deepcut], "%.3f")
            print("\nDeepCut:")
            print(f"  FLOPs: {macs_deepcut}")
            print(f"  Parameters: {params_deepcut}")
        
        # Profile Segmentation Head
        if hasattr(model, 'segmentation_head'):
            latent_size = args.image_size // 4
            dummy_features = torch.randn(1, args.vq_embed_dim, latent_size, latent_size).to(device)
            macs_seg, params_seg = profile(model.segmentation_head, inputs=(dummy_features,), verbose=False)
            macs_seg, params_seg = clever_format([macs_seg, params_seg], "%.3f")
            print("\nSegmentation Head:")
            print(f"  FLOPs: {macs_seg}")
            print(f"  Parameters: {params_seg}")
        
        # Calculate total FLOPs for forward pass
        print("\n==== TOTAL MODEL EFFICIENCY ====")
        print(f"Image Size: {args.image_size}x{args.image_size}")
        print(f"Parameters per pixel: {total_params / (args.image_size * args.image_size):.2f}")
        
    except Exception as e:
        print(f"\nError during profiling: {e}")
        print("Continuing with parameter count only...")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configure model components
    vqgan_config = {
        'n_embed': args.vq_n_embed,
        'embed_dim': args.vq_embed_dim,
        'hidden_channels': args.vq_hidden_channels,
        'n_res_blocks': args.vq_n_res_blocks,
    }
    
    gpt_config = {
        'vocab_size': args.vq_n_embed + 1,  # +1 for mask token
        'block_size': (args.image_size // 4) ** 2,
        'n_layer': args.gpt_n_layer,
        'n_head': args.gpt_n_head,
        'n_embd': args.gpt_n_embd,
    }
    
    segmentation_config = {
        'num_classes': 2,  # Binary segmentation for ISIC
        'use_deepcut': args.use_deepcut,
        'deepcut_loss_weight': args.deepcut_loss_weight,
    }
    
    # Set up ablation study flags
    use_gpt = args.ablation_mode != 'vqgan-only'
    use_mlm = args.ablation_mode == 'vqgan-gpt-mlm'
    
    print(f"\nCreating model: {args.ablation_mode}")
    print(f"  - Using GPT: {use_gpt}")
    print(f"  - Using MLM: {use_mlm}")
    
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
    
    # Load from checkpoint if specified
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        print(f"Loading model from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Profile model
    print_model_info(model, args, device)
    
    # Save results to file
    result_file = os.path.join(args.output_dir, f"model_profile_{args.ablation_mode}.txt")
    with open(result_file, 'w') as f:
        # Redirect print to file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        print_model_info(model, args, device)
        sys.stdout = original_stdout
    
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()