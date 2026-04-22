#this module handles loading the datset to use for training

from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch


def get_loaders(dataset_dir, batch_size, image_size, val_split, random_seed):

    #operations to apply to each image before entering into the model
    transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])


    #check that folder exists, raise message if it does not
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    #loads the dataset
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

    #checks that the folder is not empty
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in dataset folder: {dataset_dir}")
    
    #creates array with all class types, then finds number of classes
    class_names = full_dataset.classes
    num_classes = len(class_names)

    #splits data into training and validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size


    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    #DataLoader is a pytorch function that handles feeding data to the CNN model
    #data needs to be shuffled to train
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_names, num_classes