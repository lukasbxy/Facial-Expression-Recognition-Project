"""
Dataset Loading Utility für Facial Expression Recognition

Das Modul stellt eine Funktion bereit, um die Datasets 
automatisch als PyTorch DataLoader zu laden.

WICHTIG: Die Trainingsdaten werden automatisch mit Data Augmentation geladen:
- Random Resized Crop (90-100%)
- Random Rotation (±10°)
- Random Horizontal Flip (50%)
- Color Jitter (Helligkeit, Kontrast, Sättigung)
- Random Erasing (Cutout)

Die Validierungsdaten werden NICHT augmentiert für faire Evaluation.

Verwendung:
    from training.load_data import get_dataloaders
    train_loader, val_loader = get_dataloaders(dataset='sample')
    
Konfiguration:
    Alle Augmentation-Parameter sind in config.yaml konfigurierbar.
"""

import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


def get_dataloaders(dataset='sample', batch_size=32, num_workers=4, config_path='config.yaml', use_sampler = True):
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
    
    
    # ===================================================================
    # DATA AUGMENTATION - Training Transforms
    # ===================================================================
    # Diese Transformationen werden NUR auf die Trainingsdaten angewendet,
    # um die Datenmenge künstlich zu vergrößern und das Modell robuster
    # gegen Variationen zu machen.
    # ===================================================================
    
    aug_config = config.get('augmentation', {})
    
    train_transform = transforms.Compose([
        # 1. Random Resized Crop: Zufälliger Ausschnitt (90-100%)
        #    -> Simuliert unterschiedliche Zoom-Level und Gesichtspositionierungen
        transforms.RandomResizedCrop(
            size=img_size,
            scale=(aug_config.get('crop_scale_min', 0.9), 
                   aug_config.get('crop_scale_max', 1.0))
        ),
        
        # 2. Random Rotation: ±10 Grad
        #    -> Simuliert leicht gedrehte Köpfe/Kamerapositionen
        transforms.RandomRotation(
            degrees=aug_config.get('rotation_degrees', 10)
        ),
        
        # 3. Random Horizontal Flip: 50% Wahrscheinlichkeit
        #    -> Spiegelt Gesichter links/rechts (wichtig für Symmetrie-Invarianz)
        #    -> NICHT vertikal, da umgedrehte Gesichter unrealistisch sind
        transforms.RandomHorizontalFlip(
            p=aug_config.get('horizontal_flip_prob', 0.5)
        ),
        
        # 4. Color Jitter: Helligkeit, Kontrast, Sättigung variieren
        #    -> Simuliert unterschiedliche Lichtverhältnisse und Kameraeinstellungen
        transforms.ColorJitter(
            brightness=aug_config.get('brightness', 0.2),
            contrast=aug_config.get('contrast', 0.2),
            saturation=aug_config.get('saturation', 0.1)
        ),
        
        # 5. To Tensor: Konvertiert PIL Image zu PyTorch Tensor
        #    -> Normalisiert Pixelwerte auf [0, 1]
        transforms.ToTensor(),
        
        # 6. Random Erasing: Kleine Bereiche zufällig ausradieren (Cutout)
        #    -> Macht das Modell robust gegen Verdeckungen (z.B. Haare, Hände)
        #    -> Wird NACH ToTensor angewendet (arbeitet auf Tensoren)
        transforms.RandomErasing(
            p=aug_config.get('erase_prob', 0.3),
            scale=(aug_config.get('erase_scale_min', 0.02), 
                   aug_config.get('erase_scale_max', 0.10)),
            ratio=(0.3, 3.3)  # Seitenverhältnis der ausradierten Bereiche
        ),
    ])
    
    # ===================================================================
    # VALIDATION Transforms - KEINE Augmentation!
    # ===================================================================
    # Validierungsdaten werden NICHT augmentiert, um eine faire und
    # konsistente Evaluation zu gewährleisten.
    # ===================================================================
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    
    # Datasets mit entsprechenden Transforms laden
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    # Sampler erstellen 
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / (class_counts**0.75)
    sample_weights = class_weights[targets]
    train_sampler = WeightedRandomSampler(
        weights = torch.as_tensor(sample_weights, dtype = torch.double), # type: ignore
        num_samples = len(train_dataset),
        replacement = True
    )
    
    # DataLoader erstellen
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler = train_sampler if use_sampler else None,
        shuffle= not use_sampler,
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
