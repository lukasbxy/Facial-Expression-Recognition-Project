"""
Script to create and save confusion matrix images during training.
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import matplotlib.pyplot as plt

def create_cm(labels,
          preds,
          class_names,
          epoch: int,
          model_name: str,
          timestamp = None,
          normalize: str = "true",
          out_dir: Path = None,
          ):
    
    if out_dir is None:
        raise ValueError("out_dir must be provided")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Use timestamp if provided, otherwise use old naming scheme
    if timestamp:
        filename = f"{timestamp}_ConfusionMatrices_{model_name}_epoch_{epoch+1:03d}.png"
    else:
        filename = f"{model_name}_confusion_matrix_epoch_{epoch+1:03d}.png"
        
    out_path = out_dir / filename

    cm = confusion_matrix(labels, preds, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        xticks_rotation=45,
        colorbar=True
    )
    
    im = ax.images[0]
    im.set_clim(0.0, 1.0)
    
    ax.set_title(f"{model_name} Confusion Matrix (Epoch {epoch + 1})")
    fig.tight_layout()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path