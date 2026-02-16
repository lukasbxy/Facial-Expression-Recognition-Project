# Facial Expression Recognition Project

## Project Team
- **Paul Thiesse** - p.thiesse@campus.lmu.de ('hellany', 'THIESSE PAUL')
- **Kaan Savaş** - k.savas@campus.lmu.de ('kaiuu75')
- **Rasmus Genuit** - r.genuit@campus.lmu.de ('rg20000')
- **Lukas Boguth** - l.boguth@campus.lmu.de ('Lukas Boguth', 'lukasbxy')

## Project Overview
The goal of this project is to develop a deep learning system that automatically classifies human emotions from facial images. The system is able to:

- Detect faces in images, videos, or a live webcam stream
- Classify emotions into 6 basic categories
- Visualize the decision-making process using GradCAM (Explainable AI)

## Technical Requirements

| Requirement       | Specification                    |
|-------------------|----------------------------------|
| Training          | From scratch                     |
| Resolution        | 64x64 pixels                     |
| Channels          | 3 (RGB)                          |
| Framework         | PyTorch                          |
| Emotion Classes   | 6                                 |

## Data Augmentation

Training data is augmented with these techniques:
- Random Resized Crop (90-100%)
- Random Rotation (±10°)
- Random Horizontal Flip (50%)
- Color Jitter (Brightness, Contrast, Saturation)
- Random Erasing (2-10% of image)

All parameters are configured in `config.yaml`.

## Dataset Structure

All images use a sequential naming scheme (`000001.jpg`, `000002.jpg`, ...) per emotion class.

```
dataset/
├── Dataset_AffectNet+/
├── Dataset_FER-2013/
├── Dataset_Face-Expression-Recognition/
├── Dataset_Human-Face-Emotions/
└── Dataset_RAF_DB/
    ├── dataset_full/              # Complete dataset
    │   ├── training_set/ (80%)
    │   │   ├── 0_happiness/
    │   │   ├── 1_surprise/
    │   │   ├── 2_sadness/
    │   │   ├── 3_anger/
    │   │   ├── 4_disgust/
    │   │   └── 5_fear/
    │   └── validation_set/ (20%)
    └── dataset_sample/            # 10% subset of dataset_full
        ├── training_set/ (80%)
        └── validation_set/ (20%)
```

### Dataset Sizes (dataset_full)

| Dataset                       | Training      | Validation  | Total       |
|-------------------------------|---------------|-------------|-------------|
| AffectNet+                    | 169,621       | 42,406      | 212,027     |
| FER-2013                      | 22,386        | 5,597       | 27,983      |
| Face-Expression-Recognition   | 22,387        | 5,597       | 27,984      |
| Human-Face-Emotions           | 46,208        | 11,552      | 57,760      |
| RAF_DB                        | 9,709         | 2,426       | 12,135      |
| **Total**                     | **270,311**   | **67,578**  | **337,889** |

**Note:** Human-Face-Emotions does not contain a `4_disgust` class.

### Emotion Labels and Distribution (Training, all datasets combined)

| Index | Emotion   | Training Images | Share |
|-------|-----------|-----------------|-------|
| 0     | Happiness | 141,162         | 52.2% |
| 1     | Surprise  | 24,573          | 9.1%  |
| 2     | Sadness   | 41,947          | 15.5% |
| 3     | Anger     | 36,475          | 13.5% |
| 4     | Disgust   | 4,877           | 1.8%  |
| 5     | Fear      | 21,277          | 7.9%  |
| **-** | **Total** | **270,311**     | **100%** |

**Note:** Due to the dominance of the Happiness class (52.2%), a class limit of e.g. 50,000 images per emotion is usually applied during training to prevent bias.

**Note:** The dataset folder structure is preserved with `.gitkeep` files, but images must be downloaded separately from our Google Drive

## Training

### Setup
- `python3.12 -m venv .venv`
- macOS/Linux: `source .venv/bin/activate` | Windows: `.venv\Scripts\Activate.ps1`
- `pip install -r requirements.txt`

### How It Works
1. Choose a model with `--model` (see options below).
2. Choose datasets with `--train-datasets` and `--val-datasets` (see options below).
3. Add optional training flags (`--epochs`, `--use-scheduler`, `--use-class-weights`, etc.).
4. Run `python main.py` with the chosen CLI arguments
> training auto-selects device (`CUDA -> MPS -> CPU`), logs metrics, saves checkpoints, and applies early stopping.

### Model Options (`--model`)
| Model | Family | Default LR / WD |
|-------|--------|-----------------|
| `resnet18` | ResNet | `0.001` / `0.0001` |
| `resnet18_se` | ResNet | `0.001` / `0.0001` |
| `resnet18_variant` | ResNet | `0.001` / `0.0001` |
| `resnet18_se_variant` (default) | ResNet | `0.001` / `0.0001` |
| `resnet34` | ResNet | `0.001` / `0.0001` |
| `cct` | Transformer (CCT) | `0.0005` / `0.05` |

### Dataset Options
- Available dataset names:
  - `affectnet`
  - `fer2013`
  - `face_expression`
  - `human_emotions`
  - `raf_db`
  - `all`
- `--train-datasets` and `--val-datasets` can be combined in any way independently.
  - Use both flags together or omit both to use all datasets
- Use `--class-limit N` to cap samples per emotion across datasets.
- By default, training uses weighted random sampling (`WeightedRandomSampler`).
- Disable it with `--disable-weighted-random-sampler` (or `--disable-wrs`).

### Key Training Flags
| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | `32` | Number of epochs |
| `--lr`, `--learning-rate` | Auto | Override learning rate |
| `--weight-decay` | Auto | Override weight decay |
| `--patience` | `5` | Early stopping patience |
| `--cm-every` | `1` | Save confusion matrix every N epochs |
| `--use-scheduler` | Off | Enable OneCycleLR |
| `--use-label-smoothing` | Off | Enable label smoothing |
| `--use-class-weights` | Off | Apply class weights to the loss function (`CrossEntropyLoss`) |
| `--use-adamw` | Off | ResNet only (`cct` already uses AdamW) |
| `--disable-weighted-random-sampler`, `--disable-wrs` | Off | Disable Weighted Random Sampling for the training dataset (ResNet and CCT) |
| `--weight-power`, `--wp` | `1.0` | Controls the aggressiveness of the weighted random sampler for the training dataset. The weight is computed as $w_c = \left(\frac{1}{\mathrm{count}_c}\right)^p$ with $p$ being the weight power. Only has an effect if the weighted random sampler is enabled. |


### Example Commands
- Default:
  - `python main.py`
- ResNet with balancing options:
  - `python main.py --model resnet18 --epochs 50 --use-scheduler --use-class-weights --class-limit 50000`
- CCT with custom dataset selection:
  - `python main.py --model cct --train-datasets all --val-datasets raf_db --epochs 40 --use-scheduler`
- Mixed dataset combination example:
  - `python main.py --model resnet34 --train-datasets affectnet fer2013 human_emotions --val-datasets raf_db face_expression --epochs 40`

### Outputs
Training artifacts are saved in `runs/<ModelClass>/<timestamp>/`, including:
- `train.log`
- `metrics.csv`
- `checkpoints/best.pt` and `checkpoints/last.pt`
- `confusion_matrices/` (frequency controlled by `--cm-every`)
