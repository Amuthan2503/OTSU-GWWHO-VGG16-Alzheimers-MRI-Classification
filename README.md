# Alzheimer's Disease MRI Classification using OTSU-GWWHO Segmentation and VGG16-DCNN

This repository implements the complete pipeline of the manuscript **"Otsu Gaussian Weight Wild Horse Optimization (OTSU-GWWHO) Segmentation and Deep Learning Framework for Alzheimer's Disease Diagnosis Using MRI Images"**. It detects and classifies Alzheimer's Disease from brain MRI scans. The system combines training-only data augmentation, Otsu thresholding optimised with Gaussian Weight Wild Horse Optimization (GWWHO) for region-of-interest segmentation, and VGG16 transfer-learning classification with Grad-CAM explainability.

## Architecture Overview

The pipeline consists of five major stages:

1. **Dataset Collection and Pre-processing**: image validation, grayscale conversion, resizing to 224×224, intensity normalisation and a stratified 80:20 training–testing split
2. **Data Augmentation**: rotation, translation, adaptive gamma correction, scaling and affine transformation, applied to the training images only
3. **OTSU-GWWHO Segmentation**: Otsu within-class variance minimised by Gaussian Weight Wild Horse Optimization to obtain the optimal threshold and the brain ROI
4. **K-Fold Cross Validation**: stratified 5-fold cross-validation
5. **VGG16-DCNN Classification**: pretrained VGG16 feature extraction with a custom classification head (GAP → Dense 256 → Dropout 0.5 → Dense 128 → Dropout 0.3 → Softmax), trained with two-phase transfer learning and explained with Grad-CAM

Classes: **MD** (Mild Demented), **MOD** (Moderate Demented), **ND** (Non-Demented), **VMD** (Very Mild Demented)

## Repository Structure

| File | Description |
|------|-------------|
| `image_preprocessing.ipynb` | Image loading, validation, resizing, intensity normalisation and 80:20 split |
| `data_augmentation.ipynb` | Rotation, translation, adaptive gamma, scaling and affine augmentation (training set only) |
| `OTSU_GWWHO_segmentation.ipynb` | Otsu thresholding + Gaussian Weight Wild Horse Optimization ROI segmentation |
| `kfold_cross_validation.ipynb` | Stratified 5-fold cross-validation |
| `VGG16_DCNN_classification.ipynb` | VGG16-based DCNN with two-phase transfer learning |
| `GradCAM_explainability.ipynb` | Grad-CAM visual explanations of the predictions |
| `performance_metrics.ipynb` | Segmentation and classification metrics, 95% confidence intervals and confusion matrix |
| `runtime_classification.ipynb` | Single image inference pipeline |

## Prerequisites

- Python 3.10+
- Required libraries:

```
numpy
pandas
opencv-python
scikit-learn
matplotlib
tqdm
tensorflow
```

Install with:

```
pip install numpy pandas opencv-python scikit-learn matplotlib tqdm tensorflow
```

## Dataset

The publicly available OASIS Alzheimer's MRI dataset is used:
https://www.kaggle.com/datasets/ninadaithal/imagesoasis

The dataset should be organised as:

```
dataset/
└── Data/
    ├── Mild Dementia/
    ├── Moderate Dementia/
    ├── Non Demented/
    └── Very mild Dementia/
```

## Execution Order

Run the notebooks in the following order:

1. `image_preprocessing.ipynb`
2. `data_augmentation.ipynb`
3. `OTSU_GWWHO_segmentation.ipynb`
4. `kfold_cross_validation.ipynb`
5. `VGG16_DCNN_classification.ipynb`
6. `GradCAM_explainability.ipynb`
7. `performance_metrics.ipynb`
8. `runtime_classification.ipynb`

All intermediate files are written to the `outputs/` folder:

```
outputs/
├── 01_preprocessed/   pre-processed images and manifest
├── 02_splits/         train.csv and test.csv
├── 03_augmented/      augmented training images
├── 04_segmented/      OTSU-GWWHO ROI images, masks and thresholds
├── 05_folds/          5-fold assignment
├── 06_models/         trained VGG16-DCNN models
├── 07_results/        predictions, metrics and confusion matrix
└── 08_gradcam/        Grad-CAM visualisations
```

## Key Parameters

| Stage | Parameter | Value |
|-------|-----------|-------|
| Pre-processing | Input size | 224 × 224 |
| Pre-processing | Training / testing split | 80% / 20% |
| Augmentation | Rotation / shift / zoom / shear | ±15°, 0.1, 0.2, 0.2 |
| Augmentation | Images per method | MD ×1, MOD ×10, ND ×1, VMD ×1 |
| OTSU-GWWHO | Crossover probability PC | 0.13 |
| OTSU-GWWHO | Stallion proportion PS | 0.20 |
| OTSU-GWWHO | A_min / A_max | 0.01 / 0.01 |
| OTSU-GWWHO | Threshold range | [0, 255] |
| Cross validation | K | 5 |
| VGG16-DCNN Phase 1 | Frozen blocks / optimiser / lr / batch / epochs | Blocks 1–5 / Adam / 1e-3 / 32 / 20 |
| VGG16-DCNN Phase 2 | Frozen blocks / optimiser / lr / batch / epochs | Blocks 1–3 / Adam / 1e-5 / 32 / 30 |
| VGG16-DCNN | Loss / early stopping | Categorical cross-entropy / patience 7 |
| Grad-CAM | Target layer | `block5_conv3` |

## Evaluation Metrics

- **Segmentation**: Jaccard Similarity Coefficient (JSC), Dice Similarity Coefficient (DSC), Pixel Accuracy (PA)
- **Classification**: Precision (PPV), Sensitivity / Recall (SEN), Specificity (SPC), F1-Score, Accuracy (ACC), Error rate, 95% Confidence Intervals

## Trained Models

The trained VGG16-DCNN models are available at: **<ADD LINK>**

Place them in `outputs/06_models/` to run `GradCAM_explainability.ipynb` and `runtime_classification.ipynb` without re-training.

## License

This project is licensed under the MIT License.
