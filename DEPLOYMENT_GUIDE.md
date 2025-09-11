# Streamlit Tampered Detection App - Deployment Guide

## Quick Deployment Fix for OpenCV Error

If you're getting the `ImportError: No module named 'cv2'` error during deployment, follow these steps:

### 1. Use the Correct Requirements File

For deployment, use `requirements_deployment.txt` instead of the main `requirements.txt`:

```bash
# In your deployment platform (Streamlit Cloud, Heroku, etc.)
# Use this requirements file:
requirements_deployment.txt
```

### 2. Alternative: Update Your Main Requirements

If you must use `requirements.txt`, ensure it includes:

```
opencv-python==4.8.1.78
opencv-contrib-python==4.8.1.78
```

### 3. Streamlit Cloud Specific

For Streamlit Cloud deployment:

1. **Repository Structure**: Ensure all files are in the root directory
2. **Requirements File**: Use `requirements_deployment.txt` or update `requirements.txt`
3. **Model Files**: Ensure `objective2_robust_models/` folder is included
4. **Feature Extraction**: Ensure `objective2_feature_extraction.py` is included

### 4. File Structure for Deployment

```
your-repo/
├── streamlit_tampered_detection_app.py
├── requirements_deployment.txt
├── objective2_feature_extraction.py
├── objective2_robust_models/
│   ├── ensemble_robust.joblib
│   ├── scaler_robust.joblib
│   └── feature_info_robust.json
└── README.md
```

### 5. Deployment Commands

**For Streamlit Cloud:**
- Main file: `streamlit_tampered_detection_app.py`
- Requirements: `requirements_deployment.txt`

**For Local Testing:**
```bash
pip install -r requirements_deployment.txt
streamlit run streamlit_tampered_detection_app.py
```

### 6. Troubleshooting

**If OpenCV still fails:**
1. Check if your deployment platform supports OpenCV
2. Try using `opencv-python-headless` instead:
   ```
   opencv-python-headless==4.8.1.78
   ```

**If model files are missing:**
1. Ensure all `.joblib` files are committed to git
2. Check file paths in the app match your deployment structure

### 7. Fallback Handling

The app now includes fallback handling for OpenCV import issues. If OpenCV is not available, the app will show an error message but won't crash completely.

## Quick Fix Summary

1. **Use `requirements_deployment.txt`** for deployment
2. **Ensure all model files** are in the repository
3. **Check file paths** match your deployment structure
4. **Test locally** before deploying

This should resolve the OpenCV import error you're experiencing.
