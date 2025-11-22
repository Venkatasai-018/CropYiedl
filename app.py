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
import os
import json
from datetime import datetime
import warnings
from main import RandomForestCropRecommender
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class CropRecommendationPredictor:
    """Advanced crop recommendation predictor using Random Forest module."""
    
    def __init__(self):
        self.recommender = RandomForestCropRecommender()
        self.prediction_history = []
        self.load_model()
        
    def load_model(self):
        """Load the Random Forest model with integrated components."""
        try:
            model_path = 'random_forest_crop_recommender.pkl'
            
            if os.path.exists(model_path):
                self.recommender.load_model(model_path)
                print(f"✅ Model loaded: Random Forest")
                print(f"✅ Model accuracy: {self.recommender.accuracy:.4f}")
                return True
            elif os.path.exists('best_crop_recommendation_model.pkl'):
                print("⚠️ Old model format detected. Please run random_forest_module.py to create new model.")
                return False
            else:
                print("❌ Model file not found. Please train the Random Forest model first.")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, input_data):
        """Make crop recommendation prediction using Random Forest module."""
        if not self.recommender.is_trained:
            raise ValueError("Random Forest model not loaded")
        
        try:
            # Use the integrated Random Forest module for prediction
            result = self.recommender.predict(input_data)
            
            # Add confidence level and store in history
            confidence_level = self.get_confidence_level(result['confidence'])
            
            # Store prediction in history
            self.prediction_history.append({
                'recommended_crop': result['recommended_crop'],
                'confidence': result['confidence'],
                'confidence_level': confidence_level,
                'timestamp': datetime.now().isoformat(),
                'input_data': input_data.copy()
            })
            
            # Return formatted result
            return {
                'recommended_crop': result['recommended_crop'],
                'confidence': result['confidence'],
                'confidence_level': confidence_level,
                'top_recommendations': [
                    {
                        'crop': rec['crop'],
                        'probability': rec['probability'],
                        'confidence_level': self.get_confidence_level(rec['probability'])
                    }
                    for rec in result['top_recommendations']
                ]
            }
            
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
        try:
            if not self.prediction_history:
                return {
                    "message": "No predictions made yet",
                    'total_predictions': 0,
                    'recent_predictions': 0,
                    'average_confidence': 0,
                    'most_recommended_crop': "None",
                    'crop_distribution': {},
                    'model_name': "Random Forest Classifier" if self.recommender.is_trained else "Unknown",
                    'model_accuracy': self.recommender.accuracy if self.recommender.is_trained else 0
                }
            
            recent_predictions = self.prediction_history[-20:]
            
            crop_counts = {}
            confidence_scores = []
            
            for pred in recent_predictions:
                crop = pred.get('recommended_crop', 'Unknown')
                crop_counts[crop] = crop_counts.get(crop, 0) + 1
                
                confidence = pred.get('confidence', 0)
                if isinstance(confidence, (int, float)):
                    confidence_scores.append(confidence)
            
            return {
                'total_predictions': len(self.prediction_history),
                'recent_predictions': len(recent_predictions),
                'average_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0,
                'most_recommended_crop': max(crop_counts.items(), key=lambda x: x[1])[0] if crop_counts else "None",
                'crop_distribution': crop_counts,
                'model_name': "Random Forest Classifier" if self.recommender.is_trained else "Unknown",
                'model_accuracy': self.recommender.accuracy if self.recommender.is_trained else 0
            }
        except Exception as e:
            return {
                'error': f"Failed to get stats: {str(e)}",
                'total_predictions': 0,
                'recent_predictions': 0,
                'average_confidence': 0,
                'most_recommended_crop': "None",
                'crop_distribution': {},
                'model_name': "Random Forest Classifier" if hasattr(self, 'recommender') and self.recommender.is_trained else "Unknown",
                'model_accuracy': self.recommender.accuracy if hasattr(self, 'recommender') and self.recommender.is_trained else 0
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
    """Get model information from Random Forest module."""
    try:
        if predictor.recommender.is_trained:
            info = predictor.recommender.get_model_info()
            info['total_predictions'] = len(predictor.prediction_history)
        else:
            info = {'error': 'Random Forest model not loaded'}
        
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
    
    if predictor.recommender.is_trained:
        model_info = predictor.recommender.get_model_info()
        print(f"✅ Model: {model_info['model_name']}")
        print(f"✅ Accuracy: {predictor.recommender.accuracy:.4f}")
        print("✅ System ready!")
    else:
        print("⚠️  Model not loaded. Please run 'python random_forest_module.py' first.")
    
    print("🌐 Server starting at: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)