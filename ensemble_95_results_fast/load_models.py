#!/usr/bin/env python3
"""
Model Loader for Fast Ensemble Models
Use this to load and make predictions with saved models
"""

import pickle
import numpy as np

def load_model(model_name):
    """Load a saved ensemble model"""
    model_path = f"{model_name}_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_with_model(model, X):
    """Make predictions with loaded model"""
    return model.predict(X)

def predict_proba_with_model(model, X):
    """Get prediction probabilities with loaded model"""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    else:
        return None

# Example usage:
# model = load_model("Stacking_Top3_LR")
# predictions = predict_with_model(model, X_test)
# probabilities = predict_proba_with_model(model, X_test)
