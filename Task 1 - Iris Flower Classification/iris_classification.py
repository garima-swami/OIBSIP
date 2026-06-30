"""  
Iris Flower Classification using Machine Learning
This script trains a machine learning model to classify iris flowers
into three species: setosa, versicolor, and virginica based on their measurements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class IrisClassifier:
    """Class to handle iris flower classification"""
    
    def __init__(self, data_path):
        """Initialize the classifier with data"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load iris dataset from CSV"""
        print("Loading iris dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print("\nFirst few rows:")
        print(self.df.head())
        print("\nDataset info:")
        print(self.df.info())
        print("\nSpecies distribution:")
        print(self.df['Species'].value_counts())
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.df.describe())
        
        # Check for missing values
        print("\nMissing values:")
        print(self.df.isnull().sum())
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Sepal Length distribution by species
        for species in self.df['Species'].unique():
            axes[0, 0].hist(self.df[self.df['Species'] == species]['SepalLengthCm'], 
                           label=species, alpha=0.5)
        axes[0, 0].set_xlabel('Sepal Length (cm)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Sepal Length Distribution by Species')
        axes[0, 0].legend()
        
        # Sepal Width distribution by species
        for species in self.df['Species'].unique():
            axes[0, 1].hist(self.df[self.df['Species'] == species]['SepalWidthCm'], 
                           label=species, alpha=0.5)
        axes[0, 1].set_xlabel('Sepal Width (cm)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Sepal Width Distribution by Species')
        axes[0, 1].legend()
        
        # Petal Length distribution by species
        for species in self.df['Species'].unique():
            axes[1, 0].hist(self.df[self.df['Species'] == species]['PetalLengthCm'], 
                           label=species, alpha=0.5)
        axes[1, 0].set_xlabel('Petal Length (cm)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Petal Length Distribution by Species')
        axes[1, 0].legend()
        
        # Petal Width distribution by species
        for species in self.df['Species'].unique():
            axes[1, 1].hist(self.df[self.df['Species'] == species]['PetalWidthCm'], 
                           label=species, alpha=0.5)
        axes[1, 1].set_xlabel('Petal Width (cm)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Petal Width Distribution by Species')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('iris_distributions.png', dpi=300, bbox_inches='tight')
        print("\nDistribution plot saved as 'iris_distributions.png'")
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        correlation = self.df[features].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('iris_correlation.png', dpi=300, bbox_inches='tight')
        print("Correlation plot saved as 'iris_correlation.png'")
        plt.close()
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for training"""
        print("\n" + "="*50)
        print("DATA PREPARATION")
        print("="*50)
        
        # Extract features and target
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        X = self.df[features]
        y = self.df['Species']
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Standardize features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nTraining set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
        print(f"Features used: {features}")
        print(f"Target classes: {self.y_train.unique()}")
    
    def train_random_forest(self):
        """Train Random Forest Classifier"""
        print("\n" + "="*50)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("="*50)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Feature importance
        feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        feature_importance = self.model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importance, color='steelblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance in Random Forest Model')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved as 'feature_importance.png'")
        plt.close()
        
        print("\nFeature Importance:")
        for name, importance in zip(feature_names, feature_importance):
            print(f"  {name}: {importance:.4f}")
    
    def train_logistic_regression(self):
        """Train Logistic Regression Classifier"""
        print("\n" + "="*50)
        print("TRAINING LOGISTIC REGRESSION CLASSIFIER")
        print("="*50)
        
        self.model = LogisticRegression(
            max_iter=200,
            random_state=42,
            multi_class='multinomial'
        )
        
        self.model.fit(self.X_train, self.y_train)
        print("Logistic Regression model trained successfully!")
    
    def train_svm(self):
        """Train Support Vector Machine Classifier"""
        print("\n" + "="*50)
        print("TRAINING SUPPORT VECTOR MACHINE (SVM) CLASSIFIER")
        print("="*50)
        
        self.model = SVC(
            kernel='rbf',
            C=1.0,
            random_state=42
        )
        
        self.model.fit(self.X_train, self.y_train)
        print("SVM model trained successfully!")
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:\n{cm}")
        
        # Visualization of Confusion Matrix
        plt.figure(figsize=(10, 8))
        species = sorted(self.y_test.unique())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=species, yticklabels=species,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix plot saved as 'confusion_matrix.png'")
        plt.close()
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean()
        }
    
    def predict_single(self, measurements):
        """Predict species for new measurements"""
        measurements_scaled = self.scaler.transform([measurements])
        prediction = self.model.predict(measurements_scaled)[0]
        probability = self.model.predict_proba(measurements_scaled)[0]
        
        species_list = self.model.classes_
        
        print("\n" + "="*50)
        print("SINGLE PREDICTION")
        print("="*50)
        print(f"Input measurements: {measurements}")
        print(f"Predicted species: {prediction}")
        print("\nPrediction probabilities:")
        for species, prob in zip(species_list, probability):
            print(f"  {species}: {prob:.4f}")
        
        return prediction
    
    def run_full_pipeline(self, model_type='random_forest'):
        """Run complete classification pipeline"""
        print("\n" + "="*60)
        print("IRIS FLOWER CLASSIFICATION - FULL PIPELINE")
        print("="*60)
        
        # Load and explore data
        self.load_data()
        self.explore_data()
        
        # Prepare data
        self.prepare_data()
        
        # Train model
        if model_type == 'random_forest':
            self.train_random_forest()
        elif model_type == 'logistic_regression':
            self.train_logistic_regression()
        elif model_type == 'svm':
            self.train_svm()
        
        # Evaluate model
        metrics = self.evaluate_model()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
        return metrics


def main():
    """Main function to run the iris classification"""
    
    # Initialize classifier
    classifier = IrisClassifier('Iris.csv')
    
    # Run full pipeline with Random Forest
    print("\n>>> Training with Random Forest Classifier <<<")
    metrics = classifier.run_full_pipeline(model_type='random_forest')
    
    # Example prediction
    print("\n>>> Making a sample prediction <<<")
    # Setosa-like measurements
    sample_measurements = [5.1, 3.5, 1.4, 0.2]
    classifier.predict_single(sample_measurements)
    
    # You can also train with other models:
    # classifier.run_full_pipeline(model_type='logistic_regression')
    # classifier.run_full_pipeline(model_type='svm')


if __name__ == "__main__":
    main()
