"""
Random Forest Crop Recommendation Module
======================================

A dedicated module containing the Random Forest model with integrated
label encoder and scaler for crop recommendation predictions.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestCropRecommender:
    """
    Integrated Random Forest model for crop recommendation with built-in
    preprocessing components (scaler and label encoder).
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.crop_names = []
        self.accuracy = 0.0
        
    def load_and_prepare_data(self, data_path='Crop Recommendation dataset.csv'):
        """Load and prepare the dataset for training."""
        try:
            # Load dataset
            df = pd.read_csv(data_path)
            logger.info(f"Dataset loaded: {df.shape}")
            
            # Separate features and target
            X = df[self.feature_names]
            y = df['label']
            
            # Store unique crop names
            self.crop_names = sorted(y.unique().tolist())
            logger.info(f"Crops supported: {len(self.crop_names)} types")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def train(self, X, y, test_size=0.2):
        """Train the Random Forest model with integrated preprocessing."""
        try:
            logger.info("🌾 Starting Random Forest training...")
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Fit the scaler on training data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Fit the label encoder on training labels
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            
            # Train the Random Forest model
            self.model.fit(X_train_scaled, y_train_encoded)
            
            # Evaluate the model
            y_pred = self.model.predict(X_test_scaled)
            self.accuracy = accuracy_score(y_test_encoded, y_pred)
            
            # Mark as trained
            self.is_trained = True
            
            logger.info(f"✅ Training completed! Accuracy: {self.accuracy:.4f}")
            logger.info(f"✅ Model supports {len(self.crop_names)} crop types")
            
            # Print classification report
            print("\nClassification Report:")
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            y_test_labels = self.label_encoder.inverse_transform(y_test_encoded)
            print(classification_report(y_test_labels, y_pred_labels))
            
            return self.accuracy
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, input_data):
        """Make crop recommendation prediction."""
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        try:
            # Convert input to DataFrame if it's a dictionary
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = pd.DataFrame(input_data, columns=self.feature_names)
            
            # Scale the input
            input_scaled = self.scaler.transform(input_df)
            
            # Make prediction
            prediction_encoded = self.model.predict(input_scaled)
            probabilities = self.model.predict_proba(input_scaled)
            
            # Convert back to crop names
            predicted_crop = self.label_encoder.inverse_transform(prediction_encoded)[0]
            
            # Get confidence score
            confidence = np.max(probabilities)
            
            # Get top 3 recommendations
            top_indices = np.argsort(probabilities[0])[-3:][::-1]
            top_recommendations = []
            
            for idx in top_indices:
                crop = self.label_encoder.inverse_transform([idx])[0]
                prob = probabilities[0][idx]
                top_recommendations.append({
                    'crop': crop,
                    'probability': float(prob)
                })
            
            return {
                'recommended_crop': predicted_crop,
                'confidence': float(confidence),
                'top_recommendations': top_recommendations
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_feature_importance(self):
        """Get feature importance from the Random Forest model."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        importance_dict = {}
        for i, feature in enumerate(self.feature_names):
            importance_dict[feature] = self.model.feature_importances_[i]
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return sorted_features
    
    def save_model(self, model_path='random_forest_crop_recommender.pkl'):
        """Save the complete model with all components."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'crop_names': self.crop_names,
                'accuracy': self.accuracy,
                'is_trained': self.is_trained
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"✅ Model saved successfully: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self, model_path='random_forest_crop_recommender.pkl'):
        """Load the complete model with all components."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.crop_names = model_data['crop_names']
            self.accuracy = model_data['accuracy']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"✅ Model loaded successfully: {model_path}")
            logger.info(f"✅ Model accuracy: {self.accuracy:.4f}")
            logger.info(f"✅ Supports {len(self.crop_names)} crop types")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_model_info(self):
        """Get detailed model information."""
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        feature_importance = self.get_feature_importance()
        
        return {
            "model_name": "Random Forest Classifier",
            "accuracy": self.accuracy,
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "min_samples_split": self.model.min_samples_split,
            "supported_crops": self.crop_names,
            "feature_names": self.feature_names,
            "feature_importance": feature_importance,
            "n_features": len(self.feature_names),
            "n_classes": len(self.crop_names)
        }


def train_and_save_model(data_path='Crop Recommendation dataset.csv', 
                        model_path='random_forest_crop_recommender.pkl'):
    """Convenience function to train and save the model."""
    
    print("🌾 TRAINING RANDOM FOREST CROP RECOMMENDER 🌾")
    print("=" * 60)
    
    # Initialize the model
    recommender = RandomForestCropRecommender()
    
    # Load data
    X, y = recommender.load_and_prepare_data(data_path)
    
    # Train the model
    accuracy = recommender.train(X, y)
    
    # Display feature importance
    print("\n📊 FEATURE IMPORTANCE ANALYSIS:")
    print("-" * 40)
    feature_importance = recommender.get_feature_importance()
    for feature, importance in feature_importance:
        print(f"{feature:12}: {importance:.4f} ({importance*100:.1f}%)")
    
    # Save the model
    recommender.save_model(model_path)
    
    print(f"\n✅ Training completed successfully!")
    print(f"✅ Model accuracy: {accuracy:.4f}")
    print(f"✅ Model saved: {model_path}")
    
    return recommender


def test_model_prediction(model_path='random_forest_crop_recommender.pkl'):
    """Test the saved model with sample predictions."""
    
    print("\n🧪 TESTING MODEL PREDICTIONS")
    print("=" * 40)
    
    # Load the model
    recommender = RandomForestCropRecommender()
    recommender.load_model(model_path)
    
    # Test cases
    test_cases = [
        {
            'name': 'High Nitrogen Rice Conditions',
            'data': {'N': 90, 'P': 42, 'K': 43, 'temperature': 20.9, 
                    'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9}
        },
        {
            'name': 'Cotton Growing Conditions',
            'data': {'N': 120, 'P': 70, 'K': 40, 'temperature': 25.0,
                    'humidity': 60.0, 'ph': 7.2, 'rainfall': 100.0}
        },
        {
            'name': 'Apple Orchard Conditions',
            'data': {'N': 20, 'P': 135, 'K': 200, 'temperature': 22.0,
                    'humidity': 90.0, 'ph': 6.8, 'rainfall': 1200.0}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🌱 {test_case['name']}:")
        print(f"Input: {test_case['data']}")
        
        result = recommender.predict(test_case['data'])
        print(f"Recommended Crop: {result['recommended_crop']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("Top 3 Recommendations:")
        for i, rec in enumerate(result['top_recommendations'], 1):
            print(f"  {i}. {rec['crop']} ({rec['probability']:.3f})")


if __name__ == "__main__":
    # Train and save the model
    recommender = train_and_save_model()
    
    # Test the model
    test_model_prediction()
    
    # Display model information
    print("\n📋 MODEL INFORMATION:")
    print("=" * 40)
    info = recommender.get_model_info()
    for key, value in info.items():
        if key not in ['supported_crops', 'feature_importance']:
            print(f"{key}: {value}")