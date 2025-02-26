import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ISICDataset(Dataset):
    def __init__(self, data_dir, mask_dir, transform=None, vqtrain = False):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(data_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.vqtrain = vqtrain

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.array(Image.open(image_path).convert("RGB"))

        if not self.vqtrain:

            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            # print("Image shape:", image.shape)
            # print("Mask shape:", mask.shape)

            # Normalize mask to [0, 1]
            mask = mask / 255.0
        else:
            mask = np.array(Image.open(image_path).convert("RGB"))

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert mask to long tensor for CrossEntropyLoss
        if not self.vqtrain:
            mask = (mask > 0.5).long()

        return image, mask



def get_transforms(image_size=224):
    train_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], is_check_shapes=False)

    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_transform, val_transform

def create_dataloaders(
    train_data_dir,
    train_mask_dir,
    val_data_dir,
    val_mask_dir,
    batch_size=8,
    num_workers=4,
    image_size=224,
    vqtrain = False
):
    """Create train and validation dataloaders"""
    
    # Get transforms
    train_transform, val_transform = get_transforms(image_size)
    
    # Create datasets

    if not vqtrain:


        train_dataset = ISICDataset(
            data_dir=train_data_dir,
            mask_dir=train_mask_dir,
            transform=train_transform
        )
        
        val_dataset = ISICDataset(
            data_dir=val_data_dir,
            mask_dir=val_mask_dir,
            transform=val_transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # Important for VQGAN training
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    else:
        train_dataset = ISICDataset(
            data_dir=train_data_dir,
            mask_dir=train_mask_dir,
            transform=train_transform,
            vqtrain = True
        )

        val_dataset = ISICDataset(
            data_dir=val_data_dir,
            mask_dir=val_mask_dir,
            transform=train_transform,
            vqtrain = True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # Important for VQGAN training
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader