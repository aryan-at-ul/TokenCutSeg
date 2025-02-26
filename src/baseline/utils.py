import torch
import torchvision
# from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import albumentations as albu
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
from torch import Tensor
# from focal_loss.focal_loss import FocalLoss

def get_training_augmentation():
    # Reduced image size to lessen GPU memory usage
    train_transform = [
        # Resize to a size divisible by 32
        albu.Resize(224, 224, p=1),  # Reduced sizes, ensure they are suitable for your model
        # albu.Resize(224, 640, p=1),
        albu.PadIfNeeded(min_height=224, min_width=224, p=1),  # Adjust padding to make divisible by 32
        albu.HorizontalFlip(p=0.5),
        # Uncomment and adjust the following block if needed
        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1),
            albu.CLAHE(p=1),
            albu.HueSaturationValue(p=1)
        ], p=0.9),
        albu.GaussNoise(p=0.2),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    # Reduced image size for validation to match the training augmentation
    test_transform = [
        albu.Resize(224, 224, p=1),
        albu.PadIfNeeded(min_height=224, min_width=224, p=1)  # Adjust padding as required
    ]
    return albu.Compose(test_transform)


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

# def get_training_augmentation():
#     train_transform = [
 
#         albu.Resize(256, 416, p=1),
#         albu.PadIfNeeded(256, 416),
#         albu.HorizontalFlip(p=0.5),
 
#         albu.OneOf([
#             albu.RandomBrightnessContrast(
#                   brightness_limit=0.4, contrast_limit=0.4, p=1),
#             albu.CLAHE(p=1),
#             albu.HueSaturationValue(p=1)
#             ],
#             p=0.9,
#         ),
 
#         # albu.IAAAdditiveGaussianNoise(p=0.2),
#         albu.GaussNoise(p=0.2),
#     ]
#     return albu.Compose(train_transform)


# def get_validation_augmentation():
#     """Add paddings to make image shape divisible by 32"""
#     test_transform = [
#         albu.PadIfNeeded(256, 416)
#     ]
#     return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
 
def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
        # albu.Resize(256, 256),
    ]
    return albu.Compose(_transform)








def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()