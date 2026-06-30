# Task 1: Iris Flower Classification

## Project Overview
This task implements a machine learning classification system to classify iris flowers into three species (setosa, versicolor, and virginica) based on their physical measurements (sepal length, sepal width, petal length, and petal width).

## Dataset Information
- **Source**: Iris.csv
- **Total Samples**: 150 flowers
- **Features**: 4 (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
- **Target Classes**: 3 species (Iris-setosa, Iris-versicolor, Iris-virginica)
- **Data Split**: 80% training, 20% testing

## Project Structure
```
Task 1 - Iris Flower Classification/
├── iris_classification.py          # Main classification script
├── Iris.csv                        # Dataset
├── README.md                       # Project documentation
└── [Generated Outputs]
    ├── iris_distributions.png      # Feature distributions
    ├── iris_correlation.png        # Correlation heatmap
    ├── feature_importance.png      # Feature importance plot
    └── confusion_matrix.png        # Model performance matrix
```

## Features & Capabilities

### 1. **Data Exploration**
   - Statistical summaries of all features
   - Missing value detection
   - Distribution analysis by species
   - Feature correlation analysis

### 2. **Machine Learning Models**
   The project supports three different classification algorithms:
   - **Random Forest Classifier** (Default)
   - Logistic Regression
   - Support Vector Machine (SVM)

### 3. **Model Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Cross-validation scores
   - Confusion Matrix
   - Classification Report

### 4. **Visualizations Generated**
   - Feature distribution histograms
   - Correlation heatmap
   - Feature importance ranking
   - Confusion matrix heatmap

## How to Run

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Execute the Script
```bash
python iris_classification.py
```

The script will:
1. Load the iris dataset
2. Perform exploratory data analysis
3. Train a Random Forest classifier
4. Evaluate model performance
5. Generate visualizations
6. Make sample predictions

## Key Findings

### Model Performance
- **Accuracy**: High accuracy (typically 95%+) on test set
- **Class-wise Performance**: Well-balanced performance across all three species
- **Feature Importance**: Petal measurements are more discriminative than sepal measurements

### Data Characteristics
- Clear separation between Setosa and other species
- Some overlap between Versicolor and Virginica
- Petal length and width are strongest discriminators

## Technologies Used
- **Python 3.x**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Visualization
- **Scikit-learn**: Machine learning models

## Model Comparison

You can compare different models by uncommenting these lines in main():
```python
# Random Forest (Default)
classifier.run_full_pipeline(model_type='random_forest')

# Logistic Regression
classifier.run_full_pipeline(model_type='logistic_regression')

# Support Vector Machine
classifier.run_full_pipeline(model_type='svm')
```

## Future Improvements
- Hyperparameter tuning for optimal model performance
- Cross-validation with different K-folds
- Class weight balancing
- Feature engineering and selection
- Ensemble methods combining multiple models
- Real-time prediction API

## Author
**Created as part of OASIS INFOBYTE Internship Program**

## License
Open for educational and analytical purposes
