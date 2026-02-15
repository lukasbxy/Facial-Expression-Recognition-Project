"""
Loads datasets as PyTorch DataLoaders for training and validation.

Labels are assigned based on the emotion mapping in config.yaml, not by alphabetical
folder order (which is the ImageFolder default). This way we avoid label shifts when
a dataset is missing an emotion folder.

Training images get augmented (crop, rotation, flip, color jitter, erasing).
Validation images are not augmented.

Usage:
    from training.load_data import get_dataloaders
    train_loader, val_loader, class_names = get_dataloaders()
"""

import math
import os
import re
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image



def _load_emotion_mapping(config):
    # Read the canonical mapping from config.yaml.
    if 'emotions' not in config:
        raise ValueError(
            "No 'emotions' mapping found in config.yaml. "
            "Please add a canonical emotion mapping, e.g.:\n"
            "emotions:\n  0: happiness\n  1: surprise\n  ..."
        )
    return {int(k): v.lower() for k, v in config['emotions'].items()}


def _match_folder_to_label(folder_name, emotion_mapping):
    # Folder names are expected as "<label>_<emotion>", e.g. "0_Happiness".
    match = re.fullmatch(r"(\d+)_([A-Za-z]+)", folder_name)
    if match is None:
        return None
    
    label = int(match.group(1))
    emotion_from_folder = match.group(2).lower()
    
    if label not in emotion_mapping:
        return None
    if emotion_from_folder != emotion_mapping[label]:
        return None
    
    return label



class MappedImageFolder(Dataset):
    # Works like ImageFolder, but labels are fixed by config instead of folder sort order.
    
    def __init__(self, root, emotion_mapping, transform=None):
        self.root = root
        self.transform = transform
        self.emotion_mapping = emotion_mapping
        
        self.samples = []       # (image_path, label)
        self.targets = []       # labels only
        self.classes = []       # class names by label index
        self.class_to_idx = {}  # keeps ImageFolder-style compatibility
        self.folder_to_label = {}
        

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Build class metadata from the full mapping.
        for label in sorted(emotion_mapping.keys()):
            self.classes.append(emotion_mapping[label])
            self.class_to_idx[emotion_mapping[label]] = label
        
        # Scan folders and map them to labels.
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset path not found: {root}")
        
        folders = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])
        
        matched_emotions = []
        unmatched_folders = []
        
        for folder in folders:
            label = _match_folder_to_label(folder, emotion_mapping)
            
            if label is None:
                unmatched_folders.append(folder)
                continue
            
            self.folder_to_label[folder] = label
            matched_emotions.append(emotion_mapping[label])
            
            folder_path = os.path.join(root, folder)
            

            for fname in sorted(os.listdir(folder_path)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_extensions:
                    img_path = os.path.join(folder_path, fname)
                    self.samples.append((img_path, label))
                    self.targets.append(label)
        
        if unmatched_folders:
            raise ValueError(
                f"Folders could not be matched to any emotion in {root}: {unmatched_folders}. "
                f"Expected format '<label>_<emotion>' with canonical mapping: {emotion_mapping}"
            )
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}")
        
        # Track missing classes for logging.
        all_emotions = set(emotion_mapping.values())
        self.present_emotions = set(matched_emotions)
        self.missing_emotions = all_emotions - self.present_emotions
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_mapping_summary(self):
        # Small mapping summary for logs.
        lines = []
        for folder, label in sorted(self.folder_to_label.items(), key=lambda x: x[1]):
            emotion = self.emotion_mapping[label]
            lines.append(f"  {folder} -> {label} ({emotion})")
        
        if self.missing_emotions:
            lines.append(f"  [missing: {', '.join(sorted(self.missing_emotions))}]")
        
        return "\n".join(lines)



def _validate_class_limit(class_limit):
    if class_limit is None:
        return
    if isinstance(class_limit, bool) or not isinstance(class_limit, int) or class_limit <= 0:
        raise ValueError(f"class_limit must be a positive integer, got: {class_limit}")


def _extract_numeric_key(file_name):
    # All dataset files are expected as 6-digit numeric names, e.g. 000152.jpg.
    stem = os.path.splitext(file_name)[0]
    if re.fullmatch(r"\d{6}", stem) is None:
        return None
    return int(stem)


def _round_half_up(value):
    return int(math.floor(value + 0.5))


def _dataset_tiebreak_key(dataset_idx, dataset_names):
    return dataset_names[dataset_idx], dataset_idx


def _allocate_largest_remainder(total_to_allocate, counts_by_dataset, dataset_names):
    allocations = {dataset_idx: 0 for dataset_idx in counts_by_dataset}
    if total_to_allocate <= 0:
        return allocations
    
    total_available = sum(counts_by_dataset.values())
    if total_available <= 0:
        return allocations
    if total_to_allocate > total_available:
        raise ValueError(
            f"Cannot allocate {total_to_allocate} items when only {total_available} are available"
        )
    
    remainders = []
    allocated = 0
    for dataset_idx, count in counts_by_dataset.items():
        ideal = (total_to_allocate * count) / total_available
        base = int(math.floor(ideal))
        allocations[dataset_idx] = base
        allocated += base
        remainders.append((dataset_idx, ideal - base))
    
    leftover = total_to_allocate - allocated
    remainders.sort(
        key=lambda item: (-item[1], _dataset_tiebreak_key(item[0], dataset_names))
    )
    for dataset_idx, _ in remainders[:leftover]:
        allocations[dataset_idx] += 1
    
    return allocations


def _build_class_buckets(datasets_list):
    class_buckets = defaultdict(lambda: defaultdict(list))
    per_dataset_class_counts = [defaultdict(int) for _ in datasets_list]
    
    for dataset_idx, dataset in enumerate(datasets_list):
        for local_idx, (img_path, label) in enumerate(dataset.samples):
            file_name = os.path.basename(img_path)
            class_buckets[label][dataset_idx].append(
                {
                    "local_idx": local_idx,
                    "file_name": file_name,
                    "numeric_key": _extract_numeric_key(file_name),
                }
            )
            per_dataset_class_counts[dataset_idx][label] += 1
    
    return class_buckets, per_dataset_class_counts


def _sorted_samples_for_removal(samples):
    numeric_samples = [sample for sample in samples if sample["numeric_key"] is not None]
    non_numeric_samples = [sample for sample in samples if sample["numeric_key"] is None]
    
    numeric_samples.sort(
        key=lambda sample: (sample["numeric_key"], sample["file_name"].lower(), sample["local_idx"]),
        reverse=True
    )
    non_numeric_samples.sort(
        key=lambda sample: (sample["file_name"].lower(), sample["local_idx"]),
        reverse=True
    )
    return numeric_samples + non_numeric_samples


def _reduce_split_with_remove_targets(datasets_list, dataset_names, labels, remove_targets_by_class):
    class_buckets, before_dataset_class_counts = _build_class_buckets(datasets_list)
    removed_dataset_class_counts = [defaultdict(int) for _ in datasets_list]
    keep_indices = [set(range(len(dataset.samples))) for dataset in datasets_list]
    
    before_global = {}
    after_global = {}
    
    for label in labels:
        samples_by_dataset = class_buckets.get(label, {})
        counts_by_dataset = {
            dataset_idx: len(samples)
            for dataset_idx, samples in samples_by_dataset.items()
            if len(samples) > 0
        }
        total_before = sum(counts_by_dataset.values())
        before_global[label] = total_before
        
        remove_target = min(remove_targets_by_class.get(label, 0), total_before)
        if remove_target <= 0:
            after_global[label] = total_before
            continue
        
        remove_allocations = _allocate_largest_remainder(
            remove_target, counts_by_dataset, dataset_names
        )
        
        for dataset_idx, remove_count in remove_allocations.items():
            if remove_count <= 0:
                continue
            
            ordered_samples = _sorted_samples_for_removal(samples_by_dataset[dataset_idx])
            if remove_count > len(ordered_samples):
                raise RuntimeError(
                    f"Tried to remove {remove_count} samples but only {len(ordered_samples)} exist "
                    f"for class {label} in dataset '{dataset_names[dataset_idx]}'"
                )
            
            for sample in ordered_samples[:remove_count]:
                keep_indices[dataset_idx].discard(sample["local_idx"])
            
            removed_dataset_class_counts[dataset_idx][label] += remove_count
        
        after_global[label] = total_before - remove_target
    
    after_dataset_class_counts = [defaultdict(int) for _ in datasets_list]
    for dataset_idx in range(len(datasets_list)):
        for label in labels:
            before_count = before_dataset_class_counts[dataset_idx].get(label, 0)
            removed_count = removed_dataset_class_counts[dataset_idx].get(label, 0)
            after_count = before_count - removed_count
            if after_count > 0:
                after_dataset_class_counts[dataset_idx][label] = after_count
    
    limited_datasets = []
    for dataset_idx, dataset in enumerate(datasets_list):
        kept = sorted(keep_indices[dataset_idx])
        if len(kept) == len(dataset.samples):
            limited_datasets.append(dataset)
        else:
            limited_datasets.append(Subset(dataset, kept))
    
    return (
        limited_datasets,
        before_global,
        after_global,
        before_dataset_class_counts,
        after_dataset_class_counts
    )


def _log_global_class_limit_summary(
    split_name,
    dataset_names,
    emotion_mapping,
    before_global,
    after_global,
    before_dataset_class_counts,
    after_dataset_class_counts,
    mirrored_keep_rates=None
):
    print("\n" + "=" * 70)
    print(f"{split_name} GLOBAL CLASS LIMIT SUMMARY")
    print("=" * 70)
    
    for label in sorted(emotion_mapping.keys()):
        emotion = emotion_mapping[label]
        before_count = before_global.get(label, 0)
        after_count = after_global.get(label, 0)
        removed_count = before_count - after_count
        keep_rate = (after_count / before_count) if before_count > 0 else 1.0
        
        if mirrored_keep_rates is None:
            print(
                f"  Label {label} ({emotion}): {before_count} -> {after_count} "
                f"(removed {removed_count}, keep_rate {keep_rate:.4f})"
            )
        else:
            target_rate = mirrored_keep_rates.get(label, 1.0)
            print(
                f"  Label {label} ({emotion}): {before_count} -> {after_count} "
                f"(removed {removed_count}, keep_rate {keep_rate:.4f}, mirrored_from_train {target_rate:.4f})"
            )
    
    print("-" * 70)
    for dataset_idx, dataset_name in enumerate(dataset_names):
        changes = []
        for label in sorted(emotion_mapping.keys()):
            before_count = before_dataset_class_counts[dataset_idx].get(label, 0)
            after_count = after_dataset_class_counts[dataset_idx].get(label, 0)
            if before_count != after_count:
                changes.append(
                    f"{emotion_mapping[label]}: {before_count}->{after_count}"
                )
        
        if changes:
            print(f"  {dataset_name}: {', '.join(changes)}")
        else:
            print(f"  {dataset_name}: unchanged")
    
    print("=" * 70 + "\n")


def _apply_global_class_limit(
    train_datasets_list,
    train_dataset_names,
    val_datasets_list,
    val_dataset_names,
    emotion_mapping,
    class_limit
):
    _validate_class_limit(class_limit)
    if class_limit is None:
        return train_datasets_list, val_datasets_list
    
    labels = sorted(emotion_mapping.keys())
    
    train_totals_by_class = {label: 0 for label in labels}
    for dataset in train_datasets_list:
        for label in dataset.targets:
            train_totals_by_class[label] += 1
    
    train_remove_targets = {
        label: max(0, train_totals_by_class[label] - class_limit)
        for label in labels
    }
    
    (
        limited_train_datasets,
        train_before_global,
        train_after_global,
        train_before_dataset_class_counts,
        train_after_dataset_class_counts
    ) = _reduce_split_with_remove_targets(
        train_datasets_list,
        train_dataset_names,
        labels,
        train_remove_targets
    )
    
    train_keep_rates = {}
    for label in labels:
        before_count = train_before_global.get(label, 0)
        after_count = train_after_global.get(label, 0)
        train_keep_rates[label] = (after_count / before_count) if before_count > 0 else 1.0
    
    val_totals_by_class = {label: 0 for label in labels}
    for dataset in val_datasets_list:
        for label in dataset.targets:
            val_totals_by_class[label] += 1
    
    val_remove_targets = {}
    for label in labels:
        val_total = val_totals_by_class[label]
        if val_total == 0:
            val_remove_targets[label] = 0
            continue
        
        keep_rate = train_keep_rates[label]
        target_keep = _round_half_up(val_total * keep_rate)
        target_keep = max(1, target_keep)
        target_keep = min(val_total, target_keep)
        val_remove_targets[label] = val_total - target_keep
    
    (
        limited_val_datasets,
        val_before_global,
        val_after_global,
        val_before_dataset_class_counts,
        val_after_dataset_class_counts
    ) = _reduce_split_with_remove_targets(
        val_datasets_list,
        val_dataset_names,
        labels,
        val_remove_targets
    )
    
    _log_global_class_limit_summary(
        split_name="TRAIN",
        dataset_names=train_dataset_names,
        emotion_mapping=emotion_mapping,
        before_global=train_before_global,
        after_global=train_after_global,
        before_dataset_class_counts=train_before_dataset_class_counts,
        after_dataset_class_counts=train_after_dataset_class_counts
    )
    _log_global_class_limit_summary(
        split_name="VAL",
        dataset_names=val_dataset_names,
        emotion_mapping=emotion_mapping,
        before_global=val_before_global,
        after_global=val_after_global,
        before_dataset_class_counts=val_before_dataset_class_counts,
        after_dataset_class_counts=val_after_dataset_class_counts,
        mirrored_keep_rates=train_keep_rates
    )
    
    return limited_train_datasets, limited_val_datasets


def _load_config_and_resolve_paths(train_datasets, val_datasets, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    emotion_mapping = _load_emotion_mapping(config)
    
    if train_datasets is None:
        train_datasets = list(config['dataset'].keys())
    if val_datasets is None:
        val_datasets = list(config['dataset'].keys())
    
    train_paths = []
    val_paths = []
    
    for ds_name in train_datasets:
        if ds_name in config['dataset']:
            train_paths.append(config['dataset'][ds_name]['train'])
        else:
            raise ValueError(f"Dataset '{ds_name}' not found in config.yaml")
    
    for ds_name in val_datasets:
        if ds_name in config['dataset']:
            val_paths.append(config['dataset'][ds_name]['val'])
        else:
            raise ValueError(f"Dataset '{ds_name}' not found in config.yaml")
    
    for path in train_paths + val_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")
    
    return config, emotion_mapping, train_datasets, val_datasets, train_paths, val_paths


def _build_transforms(config):
    img_size = config['image']['size']
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
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    return train_transform, val_transform


def _load_split_datasets(train_paths, val_paths, emotion_mapping, train_transform, val_transform):
    train_datasets_list = [
        MappedImageFolder(root=path, emotion_mapping=emotion_mapping, transform=train_transform)
        for path in train_paths
    ]
    val_datasets_list = [
        MappedImageFolder(root=path, emotion_mapping=emotion_mapping, transform=val_transform)
        for path in val_paths
    ]
    return train_datasets_list, val_datasets_list


def _log_dataset_mapping(datasets_list, dataset_names, emotion_mapping):
    # Print how folders were mapped for each dataset.
    print("\n" + "=" * 70)
    print("DATASET LABEL MAPPING (from config.yaml)")
    print("=" * 70)
    

    print("Canonical mapping:")
    for label in sorted(emotion_mapping.keys()):
        print(f"  Label {label} = {emotion_mapping[label]}")
    print("-" * 70)
    

    for ds, name in zip(datasets_list, dataset_names):
        actual_ds = ds.dataset if hasattr(ds, 'dataset') else ds  # unwrap Subset
        
        if hasattr(actual_ds, 'get_mapping_summary'):
            n_classes = len(actual_ds.present_emotions)
            total_classes = len(emotion_mapping)
            n_samples = len(ds)
            
            status = "✓" if n_classes == total_classes else f"⚠ {total_classes - n_classes} missing"
            print(f"\n{name} ({n_samples} images, {n_classes}/{total_classes} classes) [{status}]")
            print(actual_ds.get_mapping_summary())
    
    print("=" * 70 + "\n")



def get_dataloaders(
    train_datasets=None, 
    val_datasets=None, 
    batch_size:int = 32, 
    num_workers:int = 4, 
    config_path:str = 'config.yaml', 
    class_limit = None,
    use_weighted_sampler: bool = True,
    weight_power: float = 1.0):
    # Build train and validation DataLoaders with fixed label mapping.
    (
        config,
        emotion_mapping,
        train_datasets,
        val_datasets,
        train_paths,
        val_paths
    ) = _load_config_and_resolve_paths(train_datasets, val_datasets, config_path)
    train_transform, val_transform = _build_transforms(config)
    
    train_datasets_list, val_datasets_list = _load_split_datasets(
        train_paths=train_paths,
        val_paths=val_paths,
        emotion_mapping=emotion_mapping,
        train_transform=train_transform,
        val_transform=val_transform
    )
    
    if class_limit is not None:
        train_datasets_list, val_datasets_list = _apply_global_class_limit(
            train_datasets_list=train_datasets_list,
            train_dataset_names=train_datasets,
            val_datasets_list=val_datasets_list,
            val_dataset_names=val_datasets,
            emotion_mapping=emotion_mapping,
            class_limit=class_limit
        )
    
    _log_dataset_mapping(
        train_datasets_list,
        [f"TRAIN: {name}" for name in train_datasets],
        emotion_mapping
    )
    _log_dataset_mapping(
        val_datasets_list,
        [f"VAL: {name}" for name in val_datasets],
        emotion_mapping
    )
    
    # Merge selected datasets into one split per phase.
    train_dataset = ConcatDataset(train_datasets_list) if len(train_datasets_list) > 1 else train_datasets_list[0]
    val_dataset = ConcatDataset(val_datasets_list) if len(val_datasets_list) > 1 else val_datasets_list[0]
    
    # Build sampler weights from the final train split (works for Subset too).
    all_targets = []
    for ds in train_datasets_list:
        if hasattr(ds, 'targets'):
            all_targets.extend(ds.targets)
        elif hasattr(ds, 'dataset') and hasattr(ds.dataset, 'targets'):

            indices = ds.indices
            original_targets = ds.dataset.targets
            all_targets.extend([original_targets[i] for i in indices])
    
    targets = np.array(all_targets)
    
    train_sampler = None
    if use_weighted_sampler and len(targets) > 0:
        class_counts = np.bincount(targets, minlength=len(emotion_mapping)).astype(np.float64)

        # avoid divide-by-zero for classes not present in the training split
        class_weights = np.zeros_like(class_counts, dtype=np.float64)
        non_zero = class_counts > 0

        # aggressiveness: (1/count)**p
        p = float(weight_power)
        class_weights[non_zero] = (1.0 / class_counts[non_zero]) ** p

        sample_weights = class_weights[targets]

        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(targets),
            replacement=True
        )
    

    train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
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
    

    class_names = [emotion_mapping[i] for i in sorted(emotion_mapping.keys())]
    
    return train_loader, val_loader, class_names


def get_datasets(train_datasets=None, val_datasets=None, batch_size=32, num_workers=4, config_path='config.yaml', class_limit=None):
    # Same setup as get_dataloaders, but returns raw datasets.
    (
        config,
        emotion_mapping,
        train_datasets,
        val_datasets,
        train_paths,
        val_paths
    ) = _load_config_and_resolve_paths(train_datasets, val_datasets, config_path)
    train_transform, val_transform = _build_transforms(config)
    
    train_datasets_list, val_datasets_list = _load_split_datasets(
        train_paths=train_paths,
        val_paths=val_paths,
        emotion_mapping=emotion_mapping,
        train_transform=train_transform,
        val_transform=val_transform
    )
    
    if class_limit is not None:
        train_datasets_list, val_datasets_list = _apply_global_class_limit(
            train_datasets_list=train_datasets_list,
            train_dataset_names=train_datasets,
            val_datasets_list=val_datasets_list,
            val_dataset_names=val_datasets,
            emotion_mapping=emotion_mapping,
            class_limit=class_limit
        )
    
    _log_dataset_mapping(
        train_datasets_list,
        [f"TRAIN: {name}" for name in train_datasets],
        emotion_mapping
    )
    _log_dataset_mapping(
        val_datasets_list,
        [f"VAL: {name}" for name in val_datasets],
        emotion_mapping
    )
    
    train_dataset = ConcatDataset(train_datasets_list) if len(train_datasets_list) > 1 else train_datasets_list[0]
    val_dataset = ConcatDataset(val_datasets_list) if len(val_datasets_list) > 1 else val_datasets_list[0]
    

    class_names = [emotion_mapping[i] for i in sorted(emotion_mapping.keys())]
    
    return train_dataset, val_dataset, class_names
