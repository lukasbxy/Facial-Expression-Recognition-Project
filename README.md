# Emotion Recognition CNN Project

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