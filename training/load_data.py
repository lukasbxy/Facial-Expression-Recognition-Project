"""
Dataset Loading Utility für Facial Expression Recognition

Das Modul stellt eine Funktion bereit, um die Datasets 
automatisch als PyTorch DataLoader zu laden.

Verwendung:
    from training.load_data import get_dataloaders
    train_loader, val_loader = get_dataloaders(dataset='sample')
"""

import os
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(dataset='sample', batch_size=32, num_workers=4, config_path='config.yaml'):
    """
    Erstellt Training- und Validation-DataLoader basierend auf der Konfiguration.
    """
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    img_size = config['image']['size']
    train_path = config['dataset'][dataset]['train']
    val_path = config['dataset'][dataset]['val']
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Dataset-Pfade nicht gefunden. Bitte config.yaml prüfen.")
    
    # Standard-Transformationen
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # Datasets laden
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    
    # DataLoader erstellen
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
