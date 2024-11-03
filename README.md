# Diabetic Retinopathy Detection

A deep learning project for detecting and classifying diabetic retinopathy using PyTorch. The project implements both multiclass (5-class) and binary classification approaches using various CNN architectures.

## Project Overview

This project aims to detect diabetic retinopathy from retinal images using different CNN architectures:
- Custom CNN
- AlexNet
- VGG

The implementation includes both multiclass classification (No_DR, Mild, Moderate, Severe, Proliferate_DR) and binary classification (No_DR vs DR) approaches.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- pandas
- numpy
- scikit-learn
- PIL (Python Imaging Library)
- matplotlib
- seaborn
- tqdm

## Dataset Structure

The dataset should be organized as follows:
```
root_path/
│
├── train.csv
└── gaussian_filtered_images/
    └── gaussian_filtered_images/
        ├── No_DR/
        ├── Mild/
        ├── Moderate/
        ├── Severe/
        └── Proliferate_DR/
```

## Features

### Multiclass Classification
- Implementation of three CNN architectures:
  - Custom CNN
  - AlexNet
  - VGG
- Data augmentation techniques
- Training with metrics tracking
- Confusion matrix visualization
- Performance evaluation using accuracy, precision, recall, and F1-score

### Binary Classification
- Custom CNN implementation
- Hyperparameter tuning
- Model evaluation
- Prediction visualization
- Confusion matrix analysis

## Model Architectures

### Custom CNN
```python
- Input Layer (224x224x3)
- Conv2D + ReLU + MaxPool2D + BatchNorm
- Conv2D + ReLU + MaxPool2D + BatchNorm
- Conv2D + ReLU + MaxPool2D + BatchNorm
- Fully Connected Layers
- Output Layer (5 classes for multiclass, 2 for binary)
```

### AlexNet
- Modified AlexNet architecture with:
  - 5 convolutional layers
  - 3 fully connected layers
  - Dropout for regularization
  - BatchNorm for better training stability

### VGG
- Modified VGG architecture with:
  - 13 convolutional layers
  - 3 fully connected layers
  - Extensive use of 3x3 convolutions
  - MaxPooling layers

## Training

### Multiclass Training
```python
python train_multiclass.py --model [custom|alexnet|vgg] --epochs 10 --batch_size 64
```

### Binary Training
```python
python train_binary.py --epochs 10 --batch_size 64
```

## Model Performance

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Hyperparameter Tuning

The binary classification model includes hyperparameter tuning for:
- Learning rates: [1e-4, 1e-3]
- Batch sizes: [32, 64]
- Number of epochs: [10, 15]
- Dropout rates: [0.1, 0.2]

## Data Preprocessing

- Image resizing to 224x224
- Normalization
- Data augmentation (for multiclass):
  - Random horizontal flip
  - Random rotation
  - Color jitter

## Model Saving and Loading

The final trained model is saved as 'diabetic_retinopathy_model.pth'.

To load the model:
```python
model = CustomCNN(dropout_rate)
model.load_state_dict(torch.load('diabetic_retinopathy_model.pth'))
```

## Visualization

The project includes visualization tools for:
- Training and validation loss curves
- Confusion matrices
- Sample predictions on test images

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.