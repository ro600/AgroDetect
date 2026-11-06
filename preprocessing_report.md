# PlantVillage preprocessing report

Generated from the prepared dataset in `dataset/`.

## 1. Dataset summary

| Metric | Value |
|---|---:|
| Classes | 38 |
| Source images discovered | 42,368 |
| Valid images processed | 42,368 |
| Corrupted images skipped | 0 |
| Output images | 42,368 |

## 2. Class distribution

| Class | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Apple___Apple_scab | 504 | 353 | 76 | 75 |
| Apple___Black_rot | 496 | 347 | 74 | 75 |
| Apple___Cedar_apple_rust | 220 | 154 | 33 | 33 |
| Apple___healthy | 1,316 | 921 | 197 | 198 |
| Blueberry___healthy | 1,202 | 841 | 180 | 181 |
| Cherry_(including_sour)___Powdery_mildew | 842 | 589 | 126 | 127 |
| Cherry_(including_sour)___healthy | 684 | 479 | 103 | 102 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 410 | 287 | 62 | 61 |
| Corn_(maize)___Common_rust_ | 953 | 667 | 143 | 143 |
| Corn_(maize)___Northern_Leaf_Blight | 788 | 552 | 118 | 118 |
| Corn_(maize)___healthy | 929 | 650 | 139 | 140 |
| Grape___Black_rot | 944 | 661 | 142 | 141 |
| Grape___Esca_(Black_Measles) | 1,107 | 775 | 166 | 166 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 861 | 603 | 129 | 129 |
| Grape___healthy | 339 | 237 | 51 | 51 |
| Orange___Haunglongbing_(Citrus_greening) | 4,405 | 3,084 | 661 | 660 |
| Peach___Bacterial_spot | 1,838 | 1,287 | 276 | 275 |
| Peach___healthy | 288 | 202 | 43 | 43 |
| Pepper,_bell___Bacterial_spot | 797 | 558 | 120 | 119 |
| Pepper,_bell___healthy | 1,182 | 828 | 177 | 177 |
| Potato___Early_blight | 800 | 560 | 120 | 120 |
| Potato___Late_blight | 800 | 560 | 120 | 120 |
| Potato___healthy | 121 | 85 | 18 | 18 |
| Raspberry___healthy | 297 | 208 | 45 | 44 |
| Soybean___healthy | 4,072 | 2,850 | 611 | 611 |
| Squash___Powdery_mildew | 1,468 | 1,028 | 220 | 220 |
| Strawberry___Leaf_scorch | 887 | 621 | 133 | 133 |
| Strawberry___healthy | 364 | 255 | 55 | 54 |
| Tomato___Bacterial_spot | 1,702 | 1,191 | 255 | 256 |
| Tomato___Early_blight | 800 | 560 | 120 | 120 |
| Tomato___Late_blight | 1,527 | 1,069 | 229 | 229 |
| Tomato___Leaf_Mold | 761 | 533 | 114 | 114 |
| Tomato___Septoria_leaf_spot | 1,417 | 992 | 213 | 212 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1,341 | 939 | 201 | 201 |
| Tomato___Target_Spot | 1,123 | 786 | 168 | 169 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 4,286 | 3,000 | 643 | 643 |
| Tomato___Tomato_mosaic_virus | 299 | 209 | 45 | 45 |
| Tomato___healthy | 197 | 138 | 30 | 29 |

The smallest class is `Potato___healthy` with 121 images and the largest
class is `Orange___Haunglongbing_(Citrus_greening)` with 4,405 images.
The class imbalance ratio is 36.40:1. Counts are stratified approximately
according to the requested 70% / 15% / 15% split.

## 3. Split statistics

| Split | Images | Percentage |
|---|---:|---:|
| Train | 29,659 | 70.00% |
| Validation | 6,356 | 15.00% |
| Test | 6,353 | 15.00% |
| **Total** | **42,368** | **100.00%** |

Each split contains all 38 class directories. The split uses random seed
`42`, and each source image is assigned to exactly one split.

## 4. Image quality and format

| Metric | Result |
|---|---|
| Output extension | `.jpg` |
| Output dimensions | `224×224` for 42,368 / 42,368 images |
| Output color mode | RGB for 42,368 / 42,368 images |
| Output size | 777,845,771 bytes |
| Average output file size | 18,359 bytes |
| JPEG encoding | Quality 90 with optimization |
| Corrupt or invalid output images | 0 |

ImageNet normalization is applied at DataLoader time rather than saved into
the image files. Training augmentation is also applied on-the-fly only to
the training split.

## 5. Data leakage checks

- The preprocessing split assigns each discovered source path once, so no
  source file is intentionally copied into multiple splits.
- Output filenames are generated independently inside each class/split and
  are not suitable for cross-split duplicate detection.
- Exact-content and perceptual near-duplicate analysis was not run; those
  checks should be added before reporting the dataset as duplicate-free.
