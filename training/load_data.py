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
import random
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms


def _limit_dataset_classes(dataset, max_samples_per_class):
    """
    Limitiert die Anzahl der Samples pro Klasse in einem Dataset.
    
    Args:
        dataset: PyTorch ImageFolder Dataset
        max_samples_per_class: Maximale Anzahl an Samples pro Klasse
    
    Returns:
        Subset mit limitierten Samples
    """
    # Class indices sammeln
    class_to_indices = {}
    for idx, (path, class_idx) in enumerate(dataset.samples):
        if class_idx not in class_to_indices:
            class_to_indices[class_idx] = []
        class_to_indices[class_idx].append(idx)
    
    # Pro Klasse zufällig samples auswählen
    selected_indices = []
    for class_idx, indices in class_to_indices.items():
        if len(indices) > max_samples_per_class:
            # Zufällige Auswahl
            selected = random.sample(indices, max_samples_per_class)
        else:
            # Alle nehmen wenn weniger als limit
            selected = indices
        selected_indices.extend(selected)
    
    print(f"Class limiting: {len(class_to_indices)} classes, "
          f"avg {max_samples_per_class} samples per class, "
          f"total {len(selected_indices)} samples")
    
    return Subset(dataset, selected_indices)


def get_dataloaders(dataset='sample', train_datasets=None, val_datasets=None, batch_size=32, num_workers=4, config_path='config.yaml', class_limit=None):
    """
    Erstellt Training- und Validation-DataLoader basierend auf der Konfiguration.
    
    Args:
        dataset: Legacy parameter für 'sample' oder 'full' (alle 5 Datasets)
        train_datasets: Liste von Dataset-Namen für Training (z.B. ['affectnet', 'fer2013'])
        val_datasets: Liste von Dataset-Namen für Validation (z.B. ['raf_db'])
        batch_size: Batch size für DataLoader
        num_workers: Anzahl der Worker für DataLoader
        config_path: Pfad zur config.yaml
        class_limit: Maximale Anzahl an Bildern pro Klasse (None für kein Limit)
    
    Returns:
        train_loader, val_loader: PyTorch DataLoader
    """
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    img_size = config['image']['size']
    
    # Dataset-Auswahl logik
    if train_datasets is None and val_datasets is None:
        # Legacy mode: 'sample' oder 'full'
        if dataset == 'full':
            # Alle 5 Datasets für Training und Validation
            train_datasets = ['affectnet', 'fer2013', 'face_expression', 'human_emotions', 'raf_db']
            val_datasets = ['affectnet', 'fer2013', 'face_expression', 'human_emotions', 'raf_db']
        elif dataset == 'sample':
            # Sample Dataset (für schnelle Tests)
            train_datasets = ['sample']
            val_datasets = ['sample']
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    # Pfade sammeln
    train_paths = []
    val_paths = []
    
    for ds_name in train_datasets:
        if ds_name in config['dataset']:
            train_paths.append(config['dataset'][ds_name]['train'])
        else:
            raise ValueError(f"Dataset '{ds_name}' nicht in config.yaml gefunden")
    
    for ds_name in val_datasets:
        if ds_name in config['dataset']:
            val_paths.append(config['dataset'][ds_name]['val'])
        else:
            raise ValueError(f"Dataset '{ds_name}' nicht in config.yaml gefunden")
    
    # Prüfen ob Pfade existieren
    for path in train_paths + val_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset-Pfad nicht gefunden: {path}")
    
    # ===================================================================
    # DATA AUGMENTATION - Training Transforms
    # ===================================================================
    
    aug_config = config.get('augmentation', {})
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=img_size,
            scale=(aug_config.get('crop_scale_min', 0.9), 
                   aug_config.get('crop_scale_max', 1.0))
        ),
        transforms.RandomRotation(
            degrees=aug_config.get('rotation_degrees', 10)
        ),
        transforms.RandomHorizontalFlip(
            p=aug_config.get('horizontal_flip_prob', 0.5)
        ),
        transforms.ColorJitter(
            brightness=aug_config.get('brightness', 0.2),
            contrast=aug_config.get('contrast', 0.2),
            saturation=aug_config.get('saturation', 0.1)
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=aug_config.get('erase_prob', 0.3),
            scale=(aug_config.get('erase_scale_min', 0.02), 
                   aug_config.get('erase_scale_max', 0.10)),
            ratio=(0.3, 3.3)
        ),
    ])
    
    # ===================================================================
    # VALIDATION Transforms - KEINE Augmentation!
    # ===================================================================
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # Datasets erstellen und kombinieren
    train_datasets_list = []
    for path in train_paths:
        dataset = datasets.ImageFolder(root=path, transform=train_transform)
        
        # Class Limiting für Training
        if class_limit is not None:
            dataset = _limit_dataset_classes(dataset, class_limit)
            print(f"Limited training dataset to {class_limit} samples per class")
        
        train_datasets_list.append(dataset)
    
    val_datasets_list = []
    for path in val_paths:
        val_datasets_list.append(datasets.ImageFolder(root=path, transform=val_transform))
    
    # Datasets kombinieren mit ConcatDataset
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets_list) if len(train_datasets_list) > 1 else train_datasets_list[0]
    val_dataset = ConcatDataset(val_datasets_list) if len(val_datasets_list) > 1 else val_datasets_list[0]
    
    # Sampler erstellen (nur für Training) - immer verwenden
    # Targets von allen Sub-Datasets sammeln (auch mit Subset)
    all_targets = []
    for ds in train_datasets_list:
        if hasattr(ds, 'targets'):
            all_targets.extend(ds.targets)
        elif hasattr(ds, 'dataset') and hasattr(ds.dataset, 'targets'):
            # Subset Fall: Original targets aus dem dataset holen
            indices = ds.indices
            original_targets = ds.dataset.targets
            all_targets.extend([original_targets[i] for i in indices])
    targets = np.array(all_targets)
    
    # Immer WeightedRandomSampler verwenden
    if len(targets) > 0:
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[targets]
        train_sampler = WeightedRandomSampler(
            weights = torch.as_tensor(sample_weights, dtype = torch.double),
            num_samples = len(targets),
            replacement = True
        )
    else:
        train_sampler = None
    
    # DataLoader erstellen
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,  # shuffle=False when using sampler
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
