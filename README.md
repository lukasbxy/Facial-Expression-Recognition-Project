# Facial Expression Recognition Project

## Project Team
- **Paul Thiesse** - p.thiesse@campus.lmu.de
- **Kaan Savaş** - k.savas@campus.lmu.de
- **Rasmus Genuit** - r.genuit@campus.lmu.de
- **Lukas Boguth** - l.boguth@campus.lmu.de

## Project Overview
The goal of this project is to develop a deep learning system that automatically classifies human emotions from facial images. The system is able to:

- Detect faces in images, videos, or a live webcam stream
- Classify emotions into 6 basic categories
- Visualize the decision-making process using Saliency Maps (Explainable AI)

## Emotion Labels

| Index | Emotion   |
|-------|-----------|
| 0     | Happiness |
| 1     | Surprise  |
| 2     | Sadness   |
| 3     | Anger     |
| 4     | Disgust   |
| 5     | Fear      |

## Technical Requirements

| Requirement       | Specification                    |
|-------------------|----------------------------------|
| Training          | From scratch                     |
| Resolution        | 64x64 pixels                     |
| Channels          | 3 (RGB)                          |
| Framework         | PyTorch                          |
| Emotion Classes   | 6 (see above)                    |

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

### Emotion Distribution (Training, all datasets combined)

| Emotion       | Images        | Share     |
|---------------|---------------|-----------|
| 0_Happiness   | 141,162       | 52.2%     |
| 1_Surprise    | 24,573        | 9.1%      |
| 2_Sadness     | 41,947        | 15.5%     |
| 3_Anger       | 36,475        | 13.5%     |
| 4_Disgust     | 4,877         | 1.8%      |
| 5_Fear        | 21,277        | 7.9%      |
| **Total**     | **270,311**   | **100%**  |

**Note:** Due to the dominance of the Happiness class (52.2%), a class limit of 50,000 images per emotion is applied during training to prevent bias.

**Note:** Only `dataset_sample` images are included in the repository. The `dataset_full` folder structure is preserved with `.gitkeep` files, but images must be downloaded separately from our Google Drive
