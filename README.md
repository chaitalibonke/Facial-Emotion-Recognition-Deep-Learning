# Facial Emotion Detection using CNN and Transfer Learning

## Overview
This project builds and evaluates multiple deep learning models to detect facial
emotions from images. Using a labeled dataset of four emotion classes, the project
progresses from custom CNN architectures to advanced transfer learning models
(VGG16, ResNet101, EfficientNetV2B2), ultimately identifying the best-performing
architecture for real-world emotion classification tasks.

---

## Problem Definition
Facial emotion recognition is a critical component of human-computer interaction,
mental health monitoring, security systems, and customer experience analytics.
This project applies computer vision and deep learning to automatically classify
facial expressions into four emotion categories using both custom and pre-trained
neural network architectures.

---

## Emotion Classes

| Class    | Description                                        |
|----------|----------------------------------------------------|
| Happy    | Smiling lips, raised cheeks, crow's feet around eyes |
| Sad      | Downturned mouth, drooping eyelids, furrowed brow  |
| Surprise | Widened eyes, raised eyebrows, open mouth          |
| Neutral  | Relaxed muscles, closed mouth, no raised brows     |

---

## Dataset

- **Structure:** Three folders — train, validation, and test
- **Classes:** happy, sad, surprise, neutral
- **Training images:** 15,109
- **Validation images:** 4,977
- **Test images:** 128
- **Class distribution (training):**

| Class    | Image Count |
|----------|-------------|
| Happy    | 3,976       |
| Sad      | 3,982       |
| Neutral  | 3,978       |
| Surprise | 3,173       |

> Note: The Surprise class has slightly fewer images than the others.
> Class imbalance is mild and was addressed through data augmentation.

---

## Objective
- Perform exploratory data analysis on facial emotion images
- Visualize class distribution and identify unique facial features per emotion
- Build and compare multiple CNN architectures from scratch
- Apply transfer learning using VGG16, ResNet101, and EfficientNetV2B2
- Evaluate all models using accuracy, loss, confusion matrix, and
  classification report
- Identify the best-performing model for deployment

---

## Methodology

### 1. Data Loading and Preprocessing
- Mount Google Drive and extract dataset from zip file
- Resize all images to 48x48 pixels
- Apply data augmentation to training set:
  - Horizontal flip
  - Brightness adjustment (range 0 to 2)
  - Rescaling (1/255)
  - Shear transformation (0.3)
- No augmentation applied to validation and test sets
- Experiment with both RGB and grayscale color modes
- Note: A similar dataset is available on Kaggle:
https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

### 2. Exploratory Data Analysis
- Visualize sample images from each emotion class in a 3x3 grid
- Identify unique facial features distinguishing each emotion
- Plot class distribution histogram across training set
- Analyze class imbalance and its potential impact on model performance

### 3. Model 1 — Base CNN (3 Convolutional Blocks)
- 3 convolutional blocks: Conv2D → MaxPooling2D → Dropout (0.2)
- Filter sizes: 64 → 32 → 32
- Fully connected: Dense(512) → Dropout(0.4) → Dense(4, softmax)
- Total parameters: 605,572
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Epochs: 20

### 4. Model 2 — Enhanced CNN (4 Convolutional Blocks + BatchNorm)
- 4 convolutional blocks: Conv2D → BatchNormalization → LeakyReLU → MaxPooling2D
- Filter sizes: 256 → 128 → 64 → 32
- Fully connected: Dense(512) → Dense(128) → Dense(4, softmax)
- Total parameters: 391,652
- Optimizer: Adam (lr=0.001)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Epochs: 20 (early stopping at epoch 19)

### 5. Transfer Learning — VGG16
- Pre-trained on ImageNet, frozen base layers
- Transfer layer: block5_pool
- Custom head: Flatten → Dense(256) → Dense(128) → Dropout(0.3) →
  Dense(64) → BatchNormalization → Dense(4, softmax)
- Input: RGB images (48x48x3)
- Epochs: 20

### 6. Transfer Learning — ResNet101
- Pre-trained on ImageNet, frozen base layers
- Transfer layer: conv5_block3_add
- Custom head: Flatten → Dense(256) → Dropout(0.3) → BatchNormalization →
  Dense(128) → Dropout(0.3) → BatchNormalization → Dense(64) →
  Dense(4, softmax)
- Input: RGB images (48x48x3)
- Epochs: 20

### 7. Transfer Learning — EfficientNetV2B2
- Pre-trained on ImageNet, frozen base layers
- Transfer layer: block6e_expand_activation
- Custom head: Flatten → Dense(256) → Dropout(0.5) → BatchNormalization →
  Dense(128) → Dropout(0.5) → BatchNormalization → Dense(4, softmax)
- Input: RGB images (48x48x3)
- Epochs: 20 (early stopping at epoch 8)

### 8. Model 3 — Complex Custom CNN (5 Convolutional Blocks, Grayscale)
- 5 convolutional blocks with increasing filter sizes: 64 → 128 → 512 → 512 → 128
- Blocks 1-3: Conv2D → ReLU → BatchNormalization → LeakyReLU →
  MaxPooling2D → Dropout(0.2)
- Blocks 4-5: Conv2D → ReLU only
- Fully connected: Dense(256) → BatchNorm → ReLU → Dropout(0.5) →
  Dense(512) → BatchNorm → ReLU → Dropout(0.5) → Dense(4, softmax)
- Input: Grayscale images (48x48x1)
- Optimizer: Adam (lr=0.003)
- Epochs: 35 (early stopping at epoch 29)

### 9. Model Evaluation
- Evaluate all models on 128 test images
- Generate confusion matrix using Seaborn heatmap
- Generate classification report with precision, recall, and F1-score
- Compare all models by test loss and test accuracy

---

## Results

| Model                    | Test Loss | Test Accuracy |
|--------------------------|-----------|---------------|
| Model 1 — Base CNN       | 0.7436    | 71.88%        |
| Model 2 — Enhanced CNN   | 0.5931    | 72.66%        |
| VGG16 Transfer Learning  | 1.0254    | 55.47%        |
| ResNet101                | 1.3893    | 25.00%        |
| EfficientNetV2B2         | 1.3913    | 25.00%        |
| Model 3 — Complex CNN    | 0.5551    | 76.56%        |

---

## Key Findings
- The custom complex CNN (Model 3) with grayscale input and 5 convolutional
  blocks achieved the best test accuracy of 76.56% and the lowest test loss
  of 0.5551
- Transfer learning models (ResNet101 and EfficientNetV2B2) performed poorly
  at 25% accuracy, likely due to the small 48x48 input size being
  incompatible with architectures designed for 224x224 images
- VGG16 performed moderately at 55.47% but was outperformed by all
  custom CNN models
- Grayscale color mode performed better than RGB for custom architectures,
  suggesting color information does not add meaningful signal for this task
  since the images are already black and white
- Data augmentation and ReduceLROnPlateau callbacks consistently improved
  model convergence and generalization
- The mild class imbalance in the Surprise class did not significantly
  impact overall model performance

---

## Model Architecture — Best Performing Model (Model 3)
```
Input (48x48x1 grayscale)
        ↓
CNN Block 1 — Conv2D(64) → ReLU → BatchNorm → LeakyReLU → MaxPool → Dropout(0.2)
        ↓
CNN Block 2 — Conv2D(128) → ReLU → BatchNorm → LeakyReLU → MaxPool → Dropout(0.2)
        ↓
CNN Block 3 — Conv2D(512) → ReLU → BatchNorm → LeakyReLU → MaxPool → Dropout(0.2)
        ↓
CNN Block 4 — Conv2D(512) → ReLU
        ↓
CNN Block 5 — Conv2D(128) → ReLU
        ↓
Flatten
        ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.5)
        ↓
Dense(512) → BatchNorm → ReLU → Dropout(0.5)
        ↓
Dense(4, softmax)

Optimizer : Adam (lr=0.003)
Loss      : Categorical Crossentropy
Epochs    : 35 (early stopping at epoch 29)
```

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV (cv2)
- Google Colab

---
## How to Run

> This notebook is designed for Google Colab. Do not run locally as google.colab is not available as a local library.

**Step 1 — Upload dataset to Google Drive:**
- Download the dataset from the course portal
- Upload `Facial_emotion_images.zip` to your Google Drive

**Step 2 — Open the notebook in Google Colab**

**Step 3 — Mount Google Drive and run all cells:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 4 — Install dependencies if needed:**
```bash
pip install tensorflow keras scikit-learn opencv-python matplotlib seaborn
```
---

## Conclusion
This project demonstrates that a well-designed custom CNN architecture trained
on grayscale images can outperform large pre-trained transfer learning models
for facial emotion recognition on small-resolution images. The complex CNN
(Model 3) with 5 convolutional blocks achieved 76.56% test accuracy, making
it the recommended model for this task. Transfer learning models such as
ResNet101 and EfficientNetV2B2 struggled due to the mismatch between their
expected input resolution and the 48x48 image size used in this dataset.
Future improvements could include fine-tuning transfer learning models with
higher resolution inputs, applying SMOTE for the Surprise class imbalance,
and exploring attention-based architectures.

---

