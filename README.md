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

| Index | Emotion   | AffectNet Label |
|-------|-----------|-----------------|
| 0     | Happiness | 1               |
| 1     | Surprise  | 3               |
| 2     | Sadness   | 2               |
| 3     | Anger     | 6               |
| 4     | Disgust   | 5               |
| 5     | Fear      | 4               |

## Technical Requirements

| Requirement       | Specification                    |
|-------------------|----------------------------------|
| Training          | From scratch                     |
| Resolution        | 64x64 pixels                     |
| Channels          | 3 (RGB)                          |
| Framework         | PyTorch                          |
| Emotion Classes   | 6 (see above)                    |

## Dataset Structure

```
dataset/
├── dataset_full/
│   ├── training_set/
│   │   ├── 0_happiness/ (134,415 images)
│   │   ├── 1_surprise/ (14,090 images)
│   │   ├── 2_sadness/ (25,459 images)
│   │   ├── 3_anger/ (24,882 images)
│   │   ├── 4_disgust/ (3,803 images)
│   │   └── 5_fear/ (6,378 images)
│   └── validation_set/ (500 images per emotion, 3,000 total)
└── dataset_sample/
    ├── training_set/ (100 images per emotion, 600 total)
    └── validation_set/ (50 images per emotion, 300 total)
```

**Note:** Only `dataset_sample` images are included in the repository. The `dataset_full` folder structure is preserved with `.gitkeep` files, but images must be downloaded separately.
