from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from pathlib import Path
import matplotlib.pyplot as plt

def create_cm(labels,
          preds,
          class_names,
          epoch:int,
          normalize: str = "true"
          ):
    
    repo_root = Path(__file__).resolve()
    while not(repo_root/"model_metrics").exists():
        repo_root = repo_root.parent
    out_dir = repo_root /"model_metrics" / "confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"confusion_matrix_epoch_{epoch+1:03d}.png"

    cm = confusion_matrix(labels, preds, normalize =normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    
    fig, ax = plt.subplots(figsize = (8,8))
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        xticks_rotation=45,
        colorbar=True,
    )
    ax.set_title(f"Confusion matrix at Epoch {epoch + 1}")
    fig.tight_layout()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
