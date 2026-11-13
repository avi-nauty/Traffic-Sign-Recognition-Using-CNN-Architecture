# GTSRB Traffic Sign Classification with a CNN

An end-to-end deep learning pipeline built with TensorFlow/Keras that classifies German traffic signs into 43 categories using the [GTSRB (German Traffic Sign Recognition Benchmark)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset.

**Test accuracy: 98.46%** on the held-out `Test.csv` set.

## Features

- **Exploratory data analysis:** parses the GTSRB training directories and plots the class distribution.
- **Efficient input pipeline:** loads, resizes (32×32) and normalizes images with `tf.data.Dataset` and `tf.data.AUTOTUNE`.
- **On-the-fly augmentation:** random rotation, translation and zoom to improve generalization.
- **Custom deep CNN:** stacked `Conv2D` blocks scaling from 64 to 512 filters, regularized with `BatchNormalization`, `Dropout` and L2 weight decay.
- **Adaptive training:** `ReduceLROnPlateau` lowers the learning rate when validation loss stalls, and `EarlyStopping` restores the best weights.
- **Evaluation and visualization:** training/validation curves, plus a grid of test images showing predicted vs. true labels.

## Workflow

1. **Reproducibility:** global random seed (`seed = 42`) set for Keras and NumPy.
2. **Dataset preparation:** image paths and integer labels are read from the directory structure, then split 80/20 into train and validation sets.
3. **Model:**
   - Input: 32×32×3 RGB images.
   - Feature extractor: two convolutional blocks (64 → 128 and 256 → 512 filters), each with `MaxPooling2D`, `BatchNormalization` and `Dropout(0.25)`.
   - Classifier head: flatten, then Dense layers (512 → 128 units) with `Dropout(0.2)`, then a 43-unit softmax output.
4. **Training:** Adam optimizer (`learning_rate=1e-3`, `ema_momentum=0.95`), `SparseCategoricalCrossentropy` loss, batch size 128.
5. **Inference:** predictions on `Test.csv`, rendered as a grid of images with predicted and ground-truth labels.

## Results

| Metric              | Score    |
|---------------------|----------|
| Training accuracy   | ~99.90%  |
| Validation accuracy | ~100.00% |
| **Test accuracy**   | **98.46%** |

> **Note:** GTSRB training images come in sequences of frames of the same physical sign. A random 80/20 split can put near-identical frames in both train and validation sets, so validation accuracy is optimistic. The test accuracy on `Test.csv` is the figure to rely on.

## Requirements

- Python 3.10+
- A GPU is recommended (the notebook was run on a Kaggle GPU kernel)

Install the dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv init
uv add tensorflow pandas numpy matplotlib seaborn scikit-learn opencv-python
```

Keras ships with TensorFlow, so it doesn't need a separate install.

## Usage

1. Download the [GTSRB dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
2. Make sure the dataset path matches `/kaggle/input/gtsrb-german-traffic-sign`, or change the path variable in the notebook to your local dataset directory.
3. Run the cells in order, either on Kaggle with a GPU enabled or locally:

```bash
uv run --with jupyterlab jupyter lab
```

## Author

Aviral Nautiyal · [GitHub](https://github.com/avi-nauty) · [LinkedIn](https://linkedin.com/in/aviral-nautiyal)

## License

Add a license (for example MIT) before publishing.
