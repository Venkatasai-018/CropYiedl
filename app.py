"""
Advanced Crop Recommendation Flask Application
===========================================

A sophisticated web application for crop recommendation using machine learning.
Features advanced UI, real-time predictions, and comprehensive analytics.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class CropRecommendationPredictor:
    """Advanced crop recommendation predictor."""
    
    def __init__(self):
        self.model_data = None
        self.prediction_history = []
        self.load_model()
        
    def load_model(self):
        """Load the trained model and preprocessing objects."""
        try:
            if os.path.exists('best_crop_recommendation_model.pkl'):
                self.model_data = joblib.load('best_crop_recommendation_model.pkl')
                print(f"✅ Model loaded: {self.model_data['model_name']}")
                print(f"✅ Model accuracy: {self.model_data['accuracy']:.4f}")
                return True
            else:
                print("❌ Model file not found. Please run advanced_analysis.py first.")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, input_data):
        """Make crop recommendation prediction."""
        if not self.model_data:
            raise ValueError("Model not loaded")
        
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            
            # Scale the features
            X_scaled = self.model_data['scaler'].transform(df)
            
            # Make prediction
            prediction = self.model_data['model'].predict(X_scaled)[0]
            prediction_proba = self.model_data['model'].predict_proba(X_scaled)[0]
            
            # Convert prediction back to crop name
            crop_name = self.model_data['label_encoder'].inverse_transform([prediction])[0]
            
            # Get confidence score
            confidence = float(np.max(prediction_proba))
            
            # Get top 3 recommendations
            top_indices = np.argsort(prediction_proba)[-3:][::-1]
            top_crops = self.model_data['label_encoder'].inverse_transform(top_indices)
            top_probabilities = prediction_proba[top_indices]
            
            recommendations = [
                {
                    'crop': crop,
                    'probability': float(prob),
                    'confidence_level': self.get_confidence_level(float(prob))
                }
                for crop, prob in zip(top_crops, top_probabilities)
            ]
            
            result = {
                'recommended_crop': crop_name,
                'confidence': confidence,
                'confidence_level': self.get_confidence_level(confidence),
                'top_recommendations': recommendations,
                'input_parameters': input_data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in history
            self.prediction_history.append(result)
            if len(self.prediction_history) > 100:  # Keep last 100 predictions
                self.prediction_history = self.prediction_history[-100:]
            
            return result
            
        except Exception as e:
            raise Exception(f"Prediction error: {e}")
    
    def get_confidence_level(self, probability):
        """Convert probability to confidence level."""
        if probability >= 0.9:
            return "Very High"
        elif probability >= 0.8:
            return "High"
        elif probability >= 0.7:
            return "Medium"
        elif probability >= 0.6:
            return "Low"
        else:
            return "Very Low"
    
    def get_stats(self):
        """Get prediction statistics."""
        if not self.prediction_history:
            return {"message": "No predictions made yet"}
        
        recent_predictions = self.prediction_history[-20:]
        
        crop_counts = {}
        confidence_scores = []
        
        for pred in recent_predictions:
            crop = pred['recommended_crop']
            crop_counts[crop] = crop_counts.get(crop, 0) + 1
            confidence_scores.append(pred['confidence'])
        
        return {
            'total_predictions': len(self.prediction_history),
            'recent_predictions': len(recent_predictions),
            'average_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0,
            'most_recommended_crop': max(crop_counts.items(), key=lambda x: x[1])[0] if crop_counts else "None",
            'crop_distribution': crop_counts,
            'model_name': self.model_data['model_name'] if self.model_data else "Unknown",
            'model_accuracy': self.model_data['accuracy'] if self.model_data else 0
        }

# Initialize predictor
predictor = CropRecommendationPredictor()

@app.route('/')
def home():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for crop prediction."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
            try:
                data[field] = float(data[field])
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': f'Invalid value for {field}'}), 400
        
        # Make prediction
        result = predictor.predict(data)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats')
def stats():
    """Get prediction statistics."""
    try:
        stats_data = predictor.get_stats()
        return jsonify({
            'success': True,
            'stats': stats_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history')
def history():
    """Get prediction history."""
    try:
        return jsonify({
            'success': True,
            'history': predictor.prediction_history[-20:]  # Last 20 predictions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-info')
def model_info():
    """Get model information."""
    try:
        if predictor.model_data:
            info = {
                'model_name': predictor.model_data['model_name'],
                'accuracy': predictor.model_data['accuracy'],
                'training_date': predictor.model_data['training_date'],
                'features': predictor.model_data['feature_names'],
                'total_predictions': len(predictor.prediction_history)
            }
        else:
            info = {'error': 'Model not loaded'}
        
        return jsonify({
            'success': True,
            'model_info': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🌾 Starting Advanced Crop Recommendation System")
    print("=" * 50)
    
    if predictor.model_data:
        print(f"✅ Model: {predictor.model_data['model_name']}")
        print(f"✅ Accuracy: {predictor.model_data['accuracy']:.4f}")
        print("✅ System ready!")
    else:
        print("⚠️  Model not loaded. Please run 'python advanced_analysis.py' first.")
    
    print("🌐 Server starting at: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)