# run_all_experiments.py
import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def parse_args():
    parser = argparse.ArgumentParser(description="Run all TokenCutSeg validation experiments")
    
    # Model checkpoints
    parser.add_argument('--vqgan-checkpoint', type=str, default="vqgan_pretrain_512/checkpoints/vqgan_pretrain_isic_512-epoch=20-val_loss_vq=0.0000.ckpt",
                        help='Path to pretrained VQGAN checkpoint')
    parser.add_argument('--segmentation-checkpoint', type=str, default="segmentation_training_2018/checkpoints/segmentation_isic2018-epoch=20-val_iou=0.0000.ckpt",
                        help='Path to trained TokenCutSeg checkpoint')
    
    parser.add_argument('--train-data-dir', type=str, default='/media/annatar/OLDHDD/isic2016/train')
    parser.add_argument('--train-mask-dir', type=str, default='/media/annatar/OLDHDD/isic2016/train_masks')
    parser.add_argument('--test-data-dir', type=str, default='/media/annatar/OLDHDD/isic2016/test')
    parser.add_argument('--test-mask-dir', type=str, default='/media/annatar/OLDHDD/isic2016/test_masks')
    
    # Experiment settings - SMALL FOR QUICK TESTING
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--max-batches', type=int, default=5,
                        help='Max batches for feature quality and graph experiments')
    parser.add_argument('--max-epochs', type=int, default=1,
                        help='Max epochs for annotation efficiency experiment')
    
    return parser.parse_args()


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_feature_quality_plot(feature_results):
    """Create clean feature quality comparison plot"""
    plt.figure(figsize=(10, 6))
    
    methods = ['ResNet-50', 'VQGAN', 'Transformer']
    values = [feature_results.get('cnn', 0), 
              feature_results.get('vqgan', 0), 
              feature_results.get('transformer', 0)]
    
    colors = ['#E74C3C', '#F39C12', '#27AE60']
    
    bars = plt.bar(methods, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement percentages
    baseline = values[0]
    vqgan_improvement = (values[1] - baseline) / baseline * 100
    transformer_improvement = (values[2] - baseline) / baseline * 100
    
    plt.text(1, values[1] + 0.05, f'+{vqgan_improvement:.1f}%', 
             ha='center', fontsize=11, fontweight='bold', color='red')
    plt.text(2, values[2] + 0.05, f'+{transformer_improvement:.1f}%', 
             ha='center', fontsize=11, fontweight='bold', color='red')
    
    plt.ylabel('mIoU', fontsize=14, fontweight='bold')
    plt.title('Feature Quality Comparison with Identical Graph-Cut Algorithm', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_quality_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('feature_quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_annotation_efficiency_plot(annotation_results):
    """Create annotation efficiency plot with actual data"""
    if not annotation_results or len(annotation_results) <= 1:
        print("Skipping annotation efficiency plot - insufficient data")
        return
    
    plt.figure(figsize=(10, 6))
    
    fractions = sorted(annotation_results.keys())
    performance = [annotation_results[f] for f in fractions]
    
    # Convert to relative performance if we have 100% baseline
    if 1.0 in annotation_results and annotation_results[1.0] > 0:
        full_perf = annotation_results[1.0]
        relative_perf = [(p / full_perf) * 100 for p in performance]
    else:
        relative_perf = [p * 100 for p in performance]
    
    x_labels = [f'{f*100:.0f}%' for f in fractions]
    
    bars = plt.bar(x_labels, relative_perf, color='#2E86AB', alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, relative_perf):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.xlabel('Percentage of Labeled Training Data', fontsize=12, fontweight='bold')
    plt.ylabel('Relative Performance (%)', fontsize=12, fontweight='bold')
    plt.title('Annotation Efficiency Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(relative_perf) * 1.1)
    
    plt.tight_layout()
    plt.savefig('annotation_efficiency.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('annotation_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_progressive_enhancement_plot(feature_results):
    """Create progressive enhancement visualization"""
    plt.figure(figsize=(10, 6))
    
    steps = ['ImageNet CNN', 'Domain-Specific VQGAN', 'Semantic Transformer']
    values = [feature_results.get('cnn', 0), 
              feature_results.get('vqgan', 0), 
              feature_results.get('transformer', 0)]
    
    x_pos = range(len(steps))
    plt.plot(x_pos, values, 'o-', linewidth=4, markersize=12, color='#2E86AB')
    plt.fill_between(x_pos, values, alpha=0.3, color='#2E86AB')
    
    # Add value annotations
    for i, (step, val) in enumerate(zip(steps, values)):
        plt.annotate(f'{val:.3f}', (i, val), xytext=(0, 15), 
                    textcoords='offset points', ha='center',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='black'))
    
    plt.xticks(x_pos, steps, fontsize=11)
    plt.ylabel('mIoU', fontsize=14, fontweight='bold')
    plt.title('Progressive Feature Enhancement', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('progressive_enhancement.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('progressive_enhancement.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_pipeline_diagram():
    """Create clean pipeline diagram"""
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.axis('off')
    
    # Pipeline steps with actual experimental context
    boxes = [
        ('Unlabeled\nDermoscopic\nImages', 0.1, '#95A5A6'),
        ('VQGAN\nEncoder', 0.25, '#F39C12'),
        ('Discrete\nTokens', 0.4, '#F39C12'),
        ('Masked Token\nPrediction', 0.55, '#27AE60'),
        ('Rich Semantic\nFeatures', 0.7, '#27AE60'),
        ('Graph-Cut\nRefinement', 0.85, '#2E86AB')
    ]
    
    box_height = 0.6
    box_width = 0.12
    
    for i, (text, x_pos, color) in enumerate(boxes):
        # Draw box
        rect = plt.Rectangle((x_pos - box_width/2, 0.2), box_width, box_height, 
                           facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x_pos, 0.5, text, ha='center', va='center', 
               fontsize=11, fontweight='bold')
        
        # Add arrow to next box
        if i < len(boxes) - 1:
            next_x = boxes[i+1][1]
            ax.annotate('', xy=(next_x - box_width/2, 0.5), 
                       xytext=(x_pos + box_width/2, 0.5),
                       arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('TokenCutSeg Architecture Pipeline', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('pipeline_diagram.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('pipeline_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_all_clean_plots(feature_results, annotation_results):
    """Create all plots separately with only real data"""
    
    print("Creating clean, separate plots with only actual experimental data...")
    
    # Plot 1: Feature Quality Comparison
    create_feature_quality_plot(feature_results)
    print("✓ Feature quality comparison plot saved")
    
    # Plot 2: Annotation Efficiency (only if we have real data)
    create_annotation_efficiency_plot(annotation_results)
    print("✓ Annotation efficiency plot saved (if data available)")
    
    # Plot 3: Progressive Enhancement
    create_progressive_enhancement_plot(feature_results)
    print("✓ Progressive enhancement plot saved")
    
    # Plot 4: Pipeline Diagram
    create_pipeline_diagram()
    print("✓ Pipeline diagram saved")
    
    print("\nAll plots saved as separate PDF and PNG files:")
    print("- feature_quality_comparison.pdf/.png")
    print("- annotation_efficiency.pdf/.png (if data available)")
    print("- progressive_enhancement.pdf/.png")
    print("- pipeline_diagram.pdf/.png")




def main():
    args = parse_args()
    
    print("=" * 60)
    print("TokenCutSeg Validation Experiments (Quick Test)")
    print("=" * 60)
    
    # Verify checkpoint paths exist
    if not os.path.exists(args.vqgan_checkpoint):
        print(f"Error: VQGAN checkpoint not found: {args.vqgan_checkpoint}")
        return
    
    if not os.path.exists(args.segmentation_checkpoint):
        print(f"Error: Segmentation checkpoint not found: {args.segmentation_checkpoint}")
        return
    
    print(f"VQGAN checkpoint: {args.vqgan_checkpoint}")
    print(f"Segmentation checkpoint: {args.segmentation_checkpoint}")
    print(f"Data directories: {args.train_data_dir}, {args.test_data_dir}")
    print(f"Quick test settings: {args.max_batches} batches, {args.max_epochs} epochs")
    
    all_results = {}
    
    # Experiment 1: Feature Quality Analysis
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: FEATURE QUALITY ANALYSIS")
    print("=" * 60)
    
    # Import and run experiment 1
    sys.path.append('.')
    
    # Run feature quality experiment
    from experiment_1_feature_quality import FeatureQualityExperiment
    from src.utils.data_loading import create_dataloaders
    
    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
        train_data_dir=args.test_data_dir,
        train_mask_dir=args.test_mask_dir,
        val_data_dir=args.test_data_dir,
        val_mask_dir=args.test_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    exp1 = FeatureQualityExperiment(args.segmentation_checkpoint, test_loader)
    feature_results = exp1.run_experiment(args.max_batches)
    all_results['feature_quality'] = feature_results
    
    print("✓ Feature quality analysis completed")
    
    # Experiment 2: Annotation Efficiency Analysis (REAL EXPERIMENT WITH SMALL DATASET)
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: ANNOTATION EFFICIENCY ANALYSIS")
    print("=" * 60)
    
    from experiment_2_annotation_efficiency import AnnotationEfficiencyExperiment
    
    exp2 = AnnotationEfficiencyExperiment(
        vqgan_checkpoint=args.vqgan_checkpoint,
        train_data_dir=args.train_data_dir,
        train_mask_dir=args.train_mask_dir,
        val_data_dir=args.test_data_dir,
        val_mask_dir=args.test_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    annotation_results = exp2.run_experiment(args.max_epochs)
    all_results['annotation_efficiency'] = annotation_results
    
    print("✓ Annotation efficiency analysis completed")
    
    # Save all results
    torch.save(all_results, 'all_experiment_results.pt')
    
    # ================================
    # CREATE AWESOME ABLATION PLOTS
    # ================================
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION-READY ABLATION PLOTS")
    print("=" * 60)
    
    # create_awesome_ablation_plots(feature_results, annotation_results)
    create_all_clean_plots(feature_results, annotation_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    if 'feature_quality' in all_results and all_results['feature_quality']:
        print("\n1. Feature Quality Analysis:")
        results = all_results['feature_quality']
        print(f"   ResNet-50 Features:   {results.get('cnn', 0):.3f}")
        print(f"   VQGAN Features:       {results.get('vqgan', 0):.3f}")
        print(f"   Transformer Features: {results.get('transformer', 0):.3f}")
        
        if results.get('cnn', 0) > 0:
            improvement = (results.get('transformer', 0) - results.get('cnn', 0)) / results.get('cnn', 1) * 100
            print(f"   Improvement over CNN: {improvement:.1f}%")
    
    if 'annotation_efficiency' in all_results and all_results['annotation_efficiency']:
        print("\n2. Annotation Efficiency:")
        results = all_results['annotation_efficiency']
        if 1.0 in results and results[1.0] > 0:
            full_perf = results[1.0]
            for fraction in [0.1, 0.25, 0.5]:
                if fraction in results:
                    relative = (results[fraction] / full_perf) * 100
                    print(f"   {fraction*100:3.0f}% annotations: {relative:.1f}% performance")
    
    print(f"\nAll results saved to: all_experiment_results.pt")
    print("\nGenerated files:")
    print("- experiment_1_feature_quality_results.pt")
    print("- experiment_2_annotation_efficiency_results.pt")
    print("- all_experiment_results.pt")
    print("- tokencutseg_comprehensive_ablation.pdf")
    print("- tokencutseg_comprehensive_ablation.png")
    
    print("\n🚀 ABLATION STUDY COMPLETE! Publication-ready plots generated.")
    print("Note: This was a quick test run with limited data.")
    print("For full publication results, increase max_batches and max_epochs.")

if __name__ == "__main__":
    main()
