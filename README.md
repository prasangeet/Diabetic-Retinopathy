# Diabetic Retinopathy Classification

A deep learning project for automated detection of diabetic retinopathy from retinal images using PyTorch. This model classifies retinal images into binary categories: presence or absence of diabetic retinopathy.

## Overview

This project implements a custom CNN architecture to analyze retinal images for signs of diabetic retinopathy. It includes data preprocessing, model training, hyperparameter tuning, and evaluation metrics.

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- kagglehub
- pandas
- numpy
- scikit-learn
- PIL (Python Imaging Library)
- matplotlib
- seaborn

## Dataset

The project uses the Diabetic Retinopathy dataset from Kaggle, specifically the preprocessed version with 224x224 Gaussian filtered images. The dataset includes:
- Multiple classes of DR severity (No_DR, Mild, Moderate, Severe, Proliferate_DR)
- Images are preprocessed and stored in class-specific folders
- Binary classification: No_DR (0) vs DR (1)

## Project Structure

1. **Data Preparation**
   - Downloads dataset using kagglehub
   - Implements custom Dataset class
   - Splits data into train/validation/test sets
   - Applies image transformations

2. **Model Architecture**
   - Custom CNN with three convolutional layers
   - Batch normalization
   - MaxPooling
   - Dropout for regularization
   - Fully connected layers

3. **Training Pipeline**
   - Cross-entropy loss
   - Adam optimizer
   - Learning rate scheduling
   - Validation metrics tracking

4. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix

5. **Hyperparameter Tuning**
   - Grid search over:
     - Learning rates: [1e-4, 1e-3]
     - Batch sizes: [32, 64]
     - Number of epochs: [10, 15]
     - Dropout rates: [0.1, 0.2]

## Usage

1. **Setup Environment**
   ```bash
   pip install torch torchvision kagglehub pandas numpy scikit-learn pillow matplotlib seaborn
   ```

2. **Download Dataset**
   ```python
   import kagglehub
   dataset_path = kagglehub.dataset_download('sovitrath/diabetic-retinopathy-224x224-gaussian-filtered')
   ```

3. **Train Model**
   ```python
   # Run training with default parameters
   model = CustomCNN(dropout_rate=0.15)
   train_model(model, train_loader, val_loader)
   
   # Or run hyperparameter tuning
   train_final_model(best_params)
   ```

4. **Save/Load Model**
   ```python
   # Save model
   torch.save(model.state_dict(), 'diabetic_retinopathy_model.pth')
   
   # Load model
   model = CustomCNN(dropout_rate=0.15)
   model.load_state_dict(torch.load('diabetic_retinopathy_model.pth'))
   ```

## Performance

The model's performance metrics include:
- Accuracy
- Precision
- Recall
- F1 Score

Actual values will vary based on hyperparameter tuning results.

## Visualization

The project includes visualization tools for:
- Confusion matrix
- Sample predictions
- Training metrics over time

## Model Architecture Details

```
CustomCNN(
  Input (3 channels) -> Conv2D (8 filters) -> ReLU -> BatchNorm -> MaxPool
  -> Conv2D (16 filters) -> ReLU -> BatchNorm -> MaxPool
  -> Conv2D (32 filters) -> ReLU -> BatchNorm -> MaxPool
  -> Flatten -> Dense (32) -> Dropout -> Dense (2)
)
```

## License

Please check the original dataset license terms on Kaggle.

## Future Improvements

- Implement data augmentation
- Try transfer learning with pre-trained models
- Add cross-validation
- Experiment with different architectures
- Implement gradient clipping
- Add early stopping

## Contributing

Feel free to submit issues and enhancement requests!
