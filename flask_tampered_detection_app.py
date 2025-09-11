#!/usr/bin/env python3
"""
Flask Application for Tampered Image Detection
Alternative deployment option with simple web interface
"""

from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import base64
import io
warnings.filterwarnings('ignore')

# Try to import OpenCV with fallback handling for deployment
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available. Some features may be limited.")

# Import our custom feature extraction classes
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class TamperedImageDetector:
    """Main class for tampered image detection using the best model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_extractor = None
        self.feature_indices = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the best performing model and preprocessing components"""
        try:
            # Load the robust ensemble model
            model_path = 'objective2_robust_models/ensemble_robust.joblib'
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print("✅ Loaded robust ensemble model (83.3% overall accuracy)")
            else:
                # Fallback to baseline model
                model_path = 'objective2_baseline_models/lightgbm.joblib'
                if os.path.exists(model_path):
                    self.model = joblib.load(model_path)
                    print("⚠️ Using baseline model")
                else:
                    print("❌ No model found")
                    return False
            
            # Load scaler
            scaler_path = 'objective2_robust_models/scaler_robust.joblib'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                scaler_path = 'objective2_baseline_models/scaler.joblib'
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
            
            # Load feature indices
            feature_info_path = 'objective2_robust_models/feature_info_robust.json'
            if os.path.exists(feature_info_path):
                with open(feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                    self.feature_indices = feature_info.get('selected_features', list(range(105)))
            else:
                feature_info_path = 'objective2_baseline_models/feature_info.json'
                if os.path.exists(feature_info_path):
                    with open(feature_info_path, 'r') as f:
                        feature_info = json.load(f)
                        self.feature_indices = feature_info.get('selected_features', list(range(105)))
            
            # Initialize feature extractor
            if CV2_AVAILABLE:
                from objective2_feature_extraction import ForensicFeatureExtractor
                self.feature_extractor = ForensicFeatureExtractor(target_size=(224, 224))
            else:
                print("Cannot initialize feature extractor without OpenCV")
                return False
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_features(self, image_path):
        """Extract features from an image"""
        try:
            features = self.feature_extractor.extract_all_features(image_path)
            
            # Convert to array and select relevant features
            feature_cols = [col for col in features.keys() if col not in ['image_path', 'label', 'tamper_type', 'category']]
            feature_values = [features[col] for col in feature_cols]
            feature_array = np.array(feature_values).reshape(1, -1)
            
            # Handle infinite and NaN values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return feature_array
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def preprocess_features(self, features):
        """Preprocess features using the same pipeline as training"""
        try:
            # Select the exact same features used during training first
            if self.feature_indices:
                features_selected = features[:, self.feature_indices]
            else:
                features_selected = features[:, :30]
            
            # Apply robust scaling
            if self.scaler:
                features_scaled = self.scaler.transform(features_selected)
            else:
                features_scaled = features_selected
            
            return features_scaled
            
        except Exception as e:
            print(f"Error preprocessing features: {e}")
            return None
    
    def predict(self, image_path):
        """Make prediction on an image"""
        if not self.is_loaded:
            return None, None, None
        
        try:
            # Extract features
            features = self.extract_features(image_path)
            if features is None:
                return None, None, None
            
            # Preprocess features
            features_processed = self.preprocess_features(features)
            if features_processed is None:
                return None, None, None
            
            # Make prediction
            prediction = self.model.predict(features_processed)[0]
            probability = self.model.predict_proba(features_processed)[0]
            
            return prediction, probability, features_processed
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None, None

# Initialize detector
detector = TamperedImageDetector()
detector.load_model()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        
        try:
            # Make prediction
            prediction, probability, features = detector.predict(temp_path)
            
            if prediction is not None:
                result = {
                    'prediction': int(prediction),
                    'confidence': {
                        'original': float(probability[0]),
                        'tampered': float(probability[1])
                    },
                    'result_text': 'TAMPERED IMAGE DETECTED' if prediction == 1 else 'ORIGINAL IMAGE',
                    'model_info': {
                        'type': 'Robust Ensemble',
                        'accuracy': '83.3%',
                        'features': 30
                    }
                }
                return jsonify(result)
            else:
                return jsonify({'error': 'Failed to analyze image'}), 500
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.is_loaded,
        'opencv_available': CV2_AVAILABLE
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create simple HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tampered Image Detection System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
            .tampered { background-color: #ffebee; border-left: 5px solid #f44336; }
            .original { background-color: #e8f5e8; border-left: 5px solid #4caf50; }
            .btn { background-color: #2196f3; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .btn:hover { background-color: #1976d2; }
        </style>
    </head>
    <body>
        <h1>🔍 Tampered Image Detection System</h1>
        <p>Advanced Forensic Image Analysis using Machine Learning</p>
        
        <div class="upload-area">
            <h3>📁 Upload Image</h3>
            <input type="file" id="fileInput" accept="image/*">
            <br><br>
            <button class="btn" onclick="analyzeImage()">🔍 Analyze Image</button>
        </div>
        
        <div id="result"></div>
        
        <h3>📊 Model Performance</h3>
        <ul>
            <li><strong>Overall Accuracy:</strong> 83.3%</li>
            <li><strong>Original Images:</strong> 66.7% accuracy</li>
            <li><strong>Tampered Images:</strong> 100% accuracy</li>
            <li><strong>Features:</strong> 30 forensic features</li>
            <li><strong>Training Data:</strong> 136 images</li>
        </ul>
        
        <script>
            async function analyzeImage() {
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) {
                    alert('Please select an image file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                resultDiv.innerHTML = '<p>Analyzing image...</p>';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        resultDiv.innerHTML = `<div class="result" style="background-color: #ffebee; color: #c62828;">❌ Error: ${data.error}</div>`;
                    } else {
                        const isTampered = data.prediction === 1;
                        const resultClass = isTampered ? 'tampered' : 'original';
                        const resultIcon = isTampered ? '🚨' : '✅';
                        
                        resultDiv.innerHTML = `
                            <div class="result ${resultClass}">
                                <h3>${resultIcon} ${data.result_text}</h3>
                                <p><strong>Confidence Scores:</strong></p>
                                <ul>
                                    <li>Original: ${(data.confidence.original * 100).toFixed(1)}%</li>
                                    <li>Tampered: ${(data.confidence.tampered * 100).toFixed(1)}%</li>
                                </ul>
                                <p><strong>Model:</strong> ${data.model_info.type} (${data.model_info.accuracy} accuracy)</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result" style="background-color: #ffebee; color: #c62828;">❌ Error: ${error.message}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(html_template)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
