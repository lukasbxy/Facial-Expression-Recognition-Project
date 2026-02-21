"""
Facial Expression Recognition Evaluation Script
This script is designed to evaluate a trained facial expression recognition model on a specified validation dataset. 

Computes:
- Overall accuracy
- Macro F1 score
- Weighted F1 score
- Detailed classification report (precision, recall, F1 per class)
- Confusion matrix (both raw and normalized)

Sample usage:
python demo/eval.py \
    --model resnet18_variant \
    --checkpoint PATH/TO/CHECKPOINT.pt \
    --val-datasets affectnet \
    --batch-size 64 \
    --output-dir results/

"""

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    f1_score,
    precision_recall_fscore_support
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.load_data import get_dataloaders
from models import ResNet18, ResNet18_SE, ResNet18_Variant, ResNet18_SE_Variant, ResNet34, CCT

def get_model(model_name: str):
    """Return a model instance for the given name."""
    models = {
        'resnet18': ResNet18,
        'resnet18_se': ResNet18_SE,
        'resnet18_se_variant': ResNet18_SE_Variant,
        'resnet34': ResNet34,
        'cct': CCT,
        'resnet18_variant': ResNet18_Variant
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name.lower()]()

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """The main evaluation loop for preds and true labels"""
    model.eval()
    
    running_loss = 0.0
    total_item = 0
    all_y_true = []
    all_y_pred = []
    
    print("Running through the validation set...")
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Calculate loss
        if criterion is not None:
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

        total_item += images.size(0)
        all_y_true.extend(labels.cpu().numpy())
        all_y_pred.extend(preds.cpu().numpy())

    avg_loss = running_loss / total_item if criterion else 0
    return avg_loss, all_y_true, all_y_pred

def plot_cm(y_true, y_pred, classes, output_path, title, normalize=False):
    """Create confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar()
    
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    """Parse and run evaluation for given model."""
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['resnet18', 'resnet18_se', 'resnet18_se_variant', 'resnet34', 'cct', 'resnet18_variant'],
        help='Select the model architecture'
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--val-datasets', type=str, nargs='+', default=['all'])
    
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        dev = torch.device('mps')
    else:
        dev = torch.device('cpu')
    print(f"Device: {dev}")

    # Load config - emotions names
    with open('config.yaml', 'r') as f: 
        config = yaml.safe_load(f)
    emotions = [config['emotions'][i] for i in range(6)]
    
    # Initialize and load specific model
    model = get_model(args.model)
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=dev)
    model.load_state_dict(checkpoint['model_state'])
    
    model.to(dev)

    # load dataloader for validation
    if 'all' in args.val_datasets:
        sets = ['affectnet', 'fer2013', 'face_expression', 'human_emotions', 'raf_db']
    else:
        sets = args.val_datasets

    _, val_dl, _ = get_dataloaders(
        train_datasets=None,
        val_datasets=sets,
        batch_size=args.batch_size
    )

    # Run eval
    loss = nn.CrossEntropyLoss()
    avg_loss, y_true, y_pred = evaluate(model, val_dl, dev, loss)

    # Compute all extra metrics
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    accuracy = np.mean(y_true_np == y_pred_np)
    macro_f1 = f1_score(y_true_np, y_pred_np, average='macro')
    weighted_f1 = f1_score(y_true_np, y_pred_np, average='weighted')
    
    print("\n" + "─"*30)
    print("     Evaluation Results")
    print("─"*30)
    
    # Print summary metrics
    print(f"Acc.:           {accuracy*100:.2f}%")
    print(f"Macro-F1:       {macro_f1:.4f}")
    print(f"Weighted-F1:    {weighted_f1:.4f}")
    print(f"Avg. Loss:      {avg_loss:.4f}")
    
    # Detailed metrics
    report = classification_report(y_true_np, y_pred_np, target_names=emotions, digits=4)
    print("\nDetailed Report:")
    print(report)

    # Output dir setup
    save_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics to text file
    with open(save_dir / "evaluation_report.txt", "w") as f:
        f.write("Evaluation Summary\n")
        f.write("-" * 20 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro-F1 Score: {macro_f1:.4f}\n")
        f.write(f"Weighted-F1 Score: {weighted_f1:.4f}\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n\n")
        f.write("Detailed Report:\n")
        f.write(report)
    
    print("Saving confusion matrices...")
    
    # Raw CM
    plot_cm(y_true_np, y_pred_np, emotions, save_dir / "confusion_matrix.png", "Confusion Matrix")
    
    # Normalized CM
    plot_cm(y_true_np, y_pred_np, emotions, save_dir / "confusion_matrix_norm.png", "Normalized Confusion Matrix", normalize=True)

    # Final bar chart for precision, recall, f1 per class
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_np, y_pred_np, labels=range(6))
    
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(len(emotions))
    plt.bar(x_axis - 0.2, precision, 0.2, label='Precision', color='skyblue')
    plt.bar(x_axis, recall, 0.2, label='Recall', color='salmon')
    plt.bar(x_axis + 0.2, f1, 0.2, label='F1-Score', color='lightgreen')
    plt.xticks(x_axis, emotions)
    plt.title("Per-Class Metrics Comparison")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_dir / "per_class_metrics.png")
    
    print(f"\nDone! Results saved to: {save_dir}")

if __name__ == '__main__':
    main()