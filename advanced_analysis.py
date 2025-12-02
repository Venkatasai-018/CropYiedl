"""
Advanced Crop Recommendation System - Data Analysis & ML Report Generator
========================================================================

This script performs comprehensive analysis of the crop recommendation dataset
and applies multiple high-level machine learning algorithms to create
detailed reports and model comparisons.

Features:
- Comprehensive Exploratory Data Analysis (EDA)
- Multiple ML algorithms comparison
- Advanced feature engineering
- Hyperparameter optimization
- Cross-validation
- Detailed performance reports
- Model interpretability analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# For advanced models
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    print("Advanced models (XGBoost, LightGBM, CatBoost) not available. Install them for full functionality.")

import os
from datetime import datetime
import joblib

class CropRecommendationAnalyzer:
    """Advanced analyzer for crop recommendation dataset."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def load_and_explore_data(self):
        """Load and perform comprehensive EDA."""
        print("="*60)
        print("CROP RECOMMENDATION DATASET ANALYSIS")
        print("="*60)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Features: {list(self.df.columns)}")
        
        # Basic info
        print("\n" + "="*40)
        print("DATASET OVERVIEW")
        print("="*40)
        print(self.df.info())
        
        print("\n" + "="*40)
        print("STATISTICAL SUMMARY")
        print("="*40)
        print(self.df.describe())
        
        # Missing values
        print("\n" + "="*40)
        print("MISSING VALUES")
        print("="*40)
        missing_values = self.df.isnull().sum()
        print(missing_values)
        
        # Target distribution
        print("\n" + "="*40)
        print("CROP DISTRIBUTION")
        print("="*40)
        crop_counts = self.df['label'].value_counts()
        print(crop_counts)
        
        print(f"\nTotal Crops: {len(crop_counts)}")
        print(f"Most Common: {crop_counts.index[0]} ({crop_counts.iloc[0]} samples)")
        print(f"Least Common: {crop_counts.index[-1]} ({crop_counts.iloc[-1]} samples)")
        
        # Create visualizations
        self.create_visualizations()
        
        return self.df
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n" + "="*40)
        print("GENERATING VISUALIZATIONS")
        print("="*40)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Correlation heatmap
        plt.subplot(3, 3, 1)
        correlation_matrix = self.df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        # 2. Target distribution
        plt.subplot(3, 3, 2)
        self.df['label'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Crop Distribution')
        plt.xticks(rotation=45)
        
        # 3. Feature distributions
        numeric_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        for i, feature in enumerate(numeric_features):
            plt.subplot(3, 3, i + 3)
            self.df[feature].hist(bins=30, alpha=0.7, color='lightgreen')
            plt.title(f'{feature} Distribution')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('crop_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as 'crop_analysis_visualizations.png'")
        
        # Additional analysis plots
        self.create_advanced_plots()
    
    def create_advanced_plots(self):
        """Create advanced analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Box plots for each feature by crop type
        features = ['N', 'P', 'K', 'temperature']
        
        for i, feature in enumerate(features):
            ax = axes[i//2, i%2]
            self.df.boxplot(column=feature, by='label', ax=ax)
            ax.set_title(f'{feature} by Crop Type')
            ax.set_xlabel('Crop')
            ax.set_ylabel(feature)
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('crop_feature_analysis.png', dpi=300, bbox_inches='tight')
        print("Advanced plots saved as 'crop_feature_analysis.png'")
    
    def prepare_data(self):
        """Prepare data for machine learning."""
        print("\n" + "="*40)
        print("DATA PREPARATION")
        print("="*40)
        
        # Separate features and target
        self.X = self.df.drop('label', axis=1)
        self.y = self.df['label']
        
        # Encode target labels
        self.y = self.label_encoder.fit_transform(self.y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print("Data scaling completed")
        
    def initialize_models(self):
        """Initialize all machine learning models."""
        print("\n" + "="*40)
        print("INITIALIZING ML MODELS")
        print("="*40)
        
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }
        
        # Add advanced models if available
        if ADVANCED_MODELS_AVAILABLE:
            self.models.update({
                'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
                'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
            })
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models."""
        print("\n" + "="*40)
        print("MODEL TRAINING & EVALUATION")
        print("="*40)
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                if name in ['Neural Network', 'SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred = model.predict(self.X_test_scaled)
                    y_pred_proba = model.predict_proba(self.X_test_scaled)
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    y_pred_proba = model.predict_proba(self.X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                # Cross-validation score
                if name in ['Neural Network', 'SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
                    cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
                else:
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
                
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                
        # Find best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['accuracy'])
        self.best_model = {
            'name': best_model_name,
            'model': self.results[best_model_name]['model'],
            'accuracy': self.results[best_model_name]['accuracy']
        }
        
        print(f"\nBest Model: {best_model_name} (Accuracy: {self.best_model['accuracy']:.4f})")
    
    def generate_comprehensive_report(self):
        """Generate detailed performance report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL PERFORMANCE REPORT")
        print("="*60)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
            'Precision': [self.results[model]['precision'] for model in self.results.keys()],
            'Recall': [self.results[model]['recall'] for model in self.results.keys()],
            'F1-Score': [self.results[model]['f1_score'] for model in self.results.keys()],
            'CV Mean': [self.results[model]['cv_mean'] for model in self.results.keys()],
            'CV Std': [self.results[model]['cv_std'] for model in self.results.keys()]
        }).sort_values('Accuracy', ascending=False)
        
        print(results_df.round(4))
        
        # Save results to CSV
        results_df.to_csv('model_performance_report.csv', index=False)
        print("\nDetailed results saved to 'model_performance_report.csv'")
        
        # Generate detailed report for best model
        best_name = self.best_model['name']
        print(f"\n" + "="*40)
        print(f"DETAILED ANALYSIS - {best_name}")
        print("="*40)
        
        # Classification report
        y_pred_best = self.results[best_name]['predictions']
        print("\nClassification Report:")
        target_names = self.label_encoder.inverse_transform(range(len(self.label_encoder.classes_)))
        print(classification_report(self.y_test, y_pred_best, 
                                  target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_best)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {best_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'confusion_matrix_best_model.png'")
        
        # Feature importance (if available)
        self.analyze_feature_importance(best_name)
        
        return results_df
    
    def analyze_feature_importance(self, model_name):
        """Analyze feature importance for tree-based models."""
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            print("\nFeature Importance Analysis:")
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(feature_importance)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.bar(feature_importance['feature'], feature_importance['importance'])
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("Feature importance plot saved as 'feature_importance.png'")
    
    def optimize_best_model(self):
        """Perform hyperparameter optimization on the best model."""
        print(f"\n" + "="*40)
        print(f"HYPERPARAMETER OPTIMIZATION - {self.best_model['name']}")
        print("="*40)
        
        best_name = self.best_model['name']
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear', 'saga'],
                'max_iter': [1000, 2000]
            }
        }
        
        if best_name in param_grids:
            model = self.models[best_name]
            param_grid = param_grids[best_name]
            
            print(f"Optimizing {best_name} with parameters: {param_grid}")
            
            # Use scaled data for models that need it
            if best_name in ['SVM', 'Neural Network', 'Logistic Regression']:
                X_train_opt = self.X_train_scaled
            else:
                X_train_opt = self.X_train
            
            # Grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_opt, self.y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Update best model
            self.best_model['model'] = grid_search.best_estimator_
            self.best_model['optimized'] = True
            
    def save_model(self):
        """Save the best trained model."""
        print("\n" + "="*40)
        print("SAVING MODEL")
        print("="*40)
        
        model_data = {
            'model': self.best_model['model'],
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': list(self.X.columns),
            'model_name': self.best_model['name'],
            'accuracy': self.best_model['accuracy'],
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, 'best_crop_recommendation_model.pkl')
        print(f"Best model ({self.best_model['name']}) saved as 'best_crop_recommendation_model.pkl'")
        
        # Save preprocessing objects separately
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
        
        return model_data
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("🌾 STARTING COMPREHENSIVE CROP RECOMMENDATION ANALYSIS 🌾")
        
        # Load and explore data
        self.load_and_explore_data()
        
        # Prepare data
        self.prepare_data()
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate
        self.train_and_evaluate_models()
        
        # Generate report
        results_df = self.generate_comprehensive_report()
        
        # Optimize best model
        self.optimize_best_model()
        
        # Save model
        model_data = self.save_model()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Best Model: {self.best_model['name']}")
        print(f"Best Accuracy: {self.best_model['accuracy']:.4f}")
        print("\nGenerated Files:")
        print("  - crop_analysis_visualizations.png")
        print("  - crop_feature_analysis.png") 
        print("  - model_performance_report.csv")
        print("  - confusion_matrix_best_model.png")
        print("  - feature_importance.png")
        print("  - best_crop_recommendation_model.pkl")
        print("  - scaler.pkl")
        print("  - label_encoder.pkl")
        print("="*60)
        
        return model_data, results_df

def main():
    """Main function to run the analysis."""
    data_path = "Crop Recommendation dataset.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset file '{data_path}' not found!")
        return None
    
    analyzer = CropRecommendationAnalyzer(data_path)
    model_data, results_df = analyzer.run_complete_analysis()
    
    return model_data, results_df

if __name__ == "__main__":
    model_data, results_df = main()