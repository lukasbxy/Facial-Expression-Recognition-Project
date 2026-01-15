from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import matplotlib.pyplot as plt

def create_cm(labels,
          preds,
          class_names,
          epoch: int,
          model_name: str,
          timestamp: str = None,
          normalize: str = "true"
          ):
    
    repo_root = Path(__file__).resolve()
    
    while not(repo_root / "model_metrics").exists():
        if repo_root == repo_root.parent:
             repo_root = Path(".").resolve()
             break 
        repo_root = repo_root.parent
        
    out_dir = repo_root / "model_metrics" / "confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Use timestamp if provided, otherwise use old naming scheme
    if timestamp:
        out_path = out_dir / f"{timestamp}_ConfusionMatrices_{model_name}_epoch_{epoch+1:03d}.png"
    else:
        out_path = out_dir / f"{model_name}_confusion_matrix_epoch_{epoch+1:03d}.png"

    cm = confusion_matrix(labels, preds, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        xticks_rotation=45,
        colorbar=True,
    )
    
    ax.set_title(f"{model_name} Confusion Matrix (Epoch {epoch + 1})")
    fig.tight_layout()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path