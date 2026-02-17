# Image Preprocessing Pipeline

## Output Format

All processed images have the following format:
- **Resolution:** 64 x 64 pixels
- **Channels:** 3 (RGB)
- **Value Range:** [0, 1] (float32)

---

## Pipeline

Simple image processing without (face detection)

**Note:** Face Detection and Eye Alignment were intentionally removed because the images are often too low-resolution or of poor quality. This led to many images where no faces were detected, which would significantly reduce the processing success rate.

| Step | Description | Parameters |
|------|-------------|------------|
| 1. Load | Load image (BGR) | - |
| 2. Grayscale → RGB | Convert to RGB (3 channels) | Gray->RGB, BGR->RGB, BGRA->RGB |
| 3. CLAHE | Contrast enhancement (lighting normalization) | clipLimit=2.0, tileGrid=8x8 |
| 4. Resize | Resize image | 64x64, INTER_AREA |
| 5. Normalize | Adjust value range | /255.0 -> [0,1] |

### Grayscale → RGB Conversion

```python
# Convert image formats to RGB with 3 channels:
if len(image.shape) == 2:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
elif image.shape[2] == 4:
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
else:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

---

## CLAHE (Contrast Limited Adaptive Histogram Equalization)

Applied to the L channel in LAB color space:

```
RGB -> LAB -> CLAHE(L) -> LAB -> RGB
```

**Parameters:**
- `clipLimit = 2.0` - Limits contrast enhancement (prevents noise)
- `tileGridSize = (8, 8)` - 8x8 regions for local adaptation

**Effect:** Normalizes different lighting conditions, improves local contrast.

---

## Configuration

```python
TARGET_SIZE = (64, 64)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
```

---

## Usage

```bash
# Single image
python data_processing/preprocess_single_image.py image.jpg

# Multiple images
python data_processing/preprocess_single_image.py img1.jpg img2.jpg img3.jpg

# Save as NumPy array
python data_processing/preprocess_single_image.py --numpy image.jpg

# Custom output directory
python data_processing/preprocess_single_image.py --output ./results image.jpg

# With visualization
python data_processing/preprocess_single_image.py --visualize image.jpg
```

---

## Dataset Structure & Splits

All datasets follow a unified structure and naming convention.

### Naming Convention

All image filenames were standardized to a sequential numbering scheme per emotion class and split:

```
000001.jpg, 000002.jpg, ..., NNNNNN.jpg
```

Images with non-standard filenames (e.g. Roboflow hashes, Pexels/Unsplash stock photo names) were removed since they were low quality and did not fit the dataset.

### Directory Structure

```
Dataset_<Name>/
├── dataset_full/          # Complete dataset
│   ├── training_set/      # 80% of images
│   │   ├── 0_happiness/
│   │   ├── 1_surprise/
│   │   ├── 2_sadness/
│   │   ├── 3_anger/
│   │   ├── 4_disgust/
│   │   └── 5_fear/
│   └── validation_set/    # 20% of images
│       └── ...
└── dataset_sample/        # 10% subset of dataset_full for quick testing and iteration
    ├── training_set/
    └── validation_set/
```

### Splits

- **Training / Validation:** 80% / 20% (per emotion class)
- **Full / Sample:** Sample = 10% of Full (per emotion class and split)
- Sample is a true subset of Full (every sample image exists identically in Full)

### Dataset Sizes (dataset_full)

#### Dataset_AffectNet+ (212,027 images)

| Emotion | Training | Validation | Total |
|---------|----------|------------|-------|
| 0_happiness | 107,932 | 26,983 | 134,915 |
| 1_surprise | 11,672 | 2,918 | 14,590 |
| 2_sadness | 20,767 | 5,192 | 25,959 |
| 3_anger | 20,306 | 5,076 | 25,382 |
| 4_disgust | 3,442 | 861 | 4,303 |
| 5_fear | 5,502 | 1,376 | 6,878 |
| **Total** | **169,621** | **42,406** | **212,027** |

#### Dataset_FER-2013 (27,983 images)

| Emotion | Training | Validation | Total |
|---------|----------|------------|-------|
| 0_Happiness | 7,038 | 1,760 | 8,798 |
| 1_Surprise | 2,602 | 650 | 3,252 |
| 2_Sadness | 4,744 | 1,186 | 5,930 |
| 3_Anger | 3,776 | 944 | 4,720 |
| 4_Disgust | 366 | 92 | 458 |
| 5_Fear | 3,860 | 965 | 4,825 |
| **Total** | **22,386** | **5,597** | **27,983** |

#### Dataset_Face-Expression-Recognition (27,984 images)

| Emotion | Training | Validation | Total |
|---------|----------|------------|-------|
| 0_Happiness | 7,040 | 1,760 | 8,800 |
| 1_Surprise | 2,601 | 650 | 3,251 |
| 2_Sadness | 4,745 | 1,186 | 5,931 |
| 3_Anger | 3,774 | 944 | 4,718 |
| 4_Disgust | 367 | 92 | 459 |
| 5_Fear | 3,860 | 965 | 4,825 |
| **Total** | **22,387** | **5,597** | **27,984** |

#### Dataset_Human-Face-Emotions (57,760 images)

| Emotion | Training | Validation | Total |
|---------|----------|------------|-------|
| 0_Happiness | 14,386 | 3,596 | 17,982 |
| 1_Surprise | 6,403 | 1,601 | 8,004 |
| 2_Sadness | 9,723 | 2,431 | 12,154 |
| 3_Anger | 7,925 | 1,981 | 9,906 |
| 5_Fear | 7,771 | 1,943 | 9,714 |
| **Total** | **46,208** | **11,552** | **57,760** |

**Note:** This dataset does not contain a `4_disgust` class.

#### Dataset_RAF_DB (12,135 images)

| Emotion | Training | Validation | Total |
|---------|----------|------------|-------|
| 0_happiness | 4,766 | 1,191 | 5,957 |
| 1_surprise | 1,295 | 324 | 1,619 |
| 2_sadness | 1,968 | 492 | 2,460 |
| 3_anger | 694 | 173 | 867 |
| 4_disgust | 702 | 175 | 877 |
| 5_fear | 284 | 71 | 355 |
| **Total** | **9,709** | **2,426** | **12,135** |

#### All Datasets Combined

| Dataset | Training | Validation | Total |
|---------|----------|------------|-------|
| AffectNet+ | 169,621 | 42,406 | 212,027 |
| FER-2013 | 22,386 | 5,597 | 27,983 |
| Face-Expression-Recognition | 22,387 | 5,597 | 27,984 |
| Human-Face-Emotions | 46,208 | 11,552 | 57,760 |
| RAF_DB | 9,709 | 2,426 | 12,135 |
| **Total** | **270,311** | **67,578** | **337,889** |
