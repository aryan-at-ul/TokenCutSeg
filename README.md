# Codebook Tokens and Deep Graph Cuts for Skin Lesion Segmentation

## Abstract
Accurate and robust skin lesion segmentation is crucial for early melanoma diagnosis and improved patient outcomes. However, current deep learning methods struggle with the inherent complexity of lesion boundaries, high inter-patient variability, and limited expert annotations. This paper introduces **TokenCutSeg**, a novel token-based skin lesion segmentation framework designed for precise boundary delineation. TokenCutSeg leverages a Vector Quantized Generative Adversarial Network (VQGAN) to map complex medical images into a compact token space, enabling efficient processing. A transformer network, trained with a self-supervised masked token prediction objective, learns rich contextual representations and effectively infers missing boundary information. Crucially, we introduce a dedicated module that refines segmentation masks using a supervised deep graph cut optimization strategy, directly addressing boundary irregularities. Evaluated on the ISIC 2016, ISIC 2017, and ISIC 2018 datasets, TokenCutSeg achieves state-of-the-art performance, demonstrating superior boundary accuracy and outperforming existing methods.



Dowload base vqgan (codebook size: 1024) [Download](https://drive.google.com/file/d/1QioCFnoYvtsq0XRXu_Xkyz6hwnDV7PVY/view?usp=drive_link)


## TokenCutSeg Results


| **Dataset** | **mIoU** | **DSC** | **Checkpoint**               |
|:-----------:|:-------:|:-------:|:----------------------------:|
| **ISIC16**  | **86.18** | **93.30** | [Download](https://drive.google.com/file/d/1WJZynHSSCl8yCh9Vf3-dogPHfixZ5Wa1/view?usp=drive_link)       |
| **ISIC17**  | **81.37** | **90.71** | [Download](https://drive.google.com/file/d/1lb7oPumyzE9mmrg6YApeJ-dTXU0vMK67/view?usp=drive_link)      |
| **ISIC18**  | **82.76** | **90.39** | [Download](https://drive.google.com/file/d/1WJZynHSSCl8yCh9Vf3-dogPHfixZ5Wa1/view?usp=drive_link)     |
| **16 → 17** | **74.33** | **87.19** | same as isic2016     |
| **16 → 18** | **80.40** | **88.92** | same as isic2016     |


## Step 1: Train the VQGAN Model

Use the `train_vq.py` script to train VQGAN on a large corpus of images. Below is the list of default arguments included in the script:

```
python train_vq.py \
    --train-data-dir /path/to/train/images \
    --train-mask-dir /path/to/train/masks \
    --val-data-dir /path/to/val/images \
    --val-mask-dir /path/to/val/masks \ #val and train are the same (todo) lowest req loss can be saved
    --batch-size 4 \
    --num-workers 4 \
    --learning-rate 1e-4 \
    --max-epochs 100 \
    --gpus 1 \
    --precision 32 \
    --vq-embed-dim 256 \
    --vq-n-embed 64 \
    --vq-hidden-channels 128 \
    --vq-n-res-blocks 2 \
    --disc-start 10000 \
    --disc-weight 0.5 \
    --perceptual-weight 0.8 \
    --codebook-weight 1.0 \
    --output-dir "vqgan_pretrain_full" \
    --experiment-name "vqgan_pretrain_isic_full"

```

## Step 2: Train the Segmentation Model

Use the `train_seg.py` script to train the **TokenCutSeg** segmentation model. Below is the list of default arguments included in the script:

```
python train_seg.py \
    --train-data-dir /path/to/train/images \
    --train-mask-dir /path/to/train/masks \
    --val-data-dir /path/to/val/images \
    --val-mask-dir /path/to/val/masks \
    --image-size 224 \
    --batch-size 1 \
    --num-workers 4 \
    --learning-rate 1e-4 \
    --max-epochs 200 \
    --gpus 1 \
    --precision 16 \
    --vqgan-checkpoint "model_trained_in_step1.ckpt" \
    --vq-embed-dim 256 \
    --vq-n-embed 512 \
    --vq-hidden-channels 128 \
    --vq-n-res-blocks 2 \
    --gpt-n-layer 12 \
    --gpt-n-head 4 \
    --gpt-n-embd 256 \
    --use-deepcut True \
    --deepcut-loss-weight 0.2 \
    --output-dir "segmentation_training_2018_codebook_full" \
    --experiment-name "segmentation_isic2018_codebook_full"

```

## Step 3: Test the Segmentation Model

Use the `test_seg.py` script to evaluate the trained **TokenCutSeg** model on the test data. Below is the list of default arguments included in the script:

```
python test_seg.py \
    --test-data-dir /path/to/test/images \
    --test-mask-dir /path/to/test/masks \
    --image-size 224 \
    --checkpoint-path "model_trained_in_step2.ckpt" \
    --batch-size 1 \
    --num-workers 4 \
    --vq-embed-dim 256 \
    --vq-n-embed 1024 \
    --vq-hidden-channels 128 \
    --vq-n-res-blocks 2 \
    --gpt-n-layer 12 \
    --gpt-n-head 4 \
    --gpt-n-embd 256

```

To replicate our **TokenCutSeg** results, simply leave all arguments at their default values and download the provided checkpoints (table above).


Ablation study checkpoints can be found here [savedmodels](https://drive.google.com/drive/folders/11_9iTLvHM3vQkgAyU871yUZ1fHzhh-Aw?usp=drive_link)
