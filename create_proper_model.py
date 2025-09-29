#!/usr/bin/env python3
"""
Create proper ensemble model with predict_proba for Streamlit app
"""

import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import json

def create_proper_model():
    print('Loading features...')
    features_df = pd.read_csv('new_objective2_features/feature_vectors/comprehensive_features.csv')
    feature_columns = [col for col in features_df.columns 
                      if col not in ['image_path', 'label', 'tamper_type', 'category', 'has_mask']]

    X_features = features_df[feature_columns].values
    y_features = features_df['label'].values

    print(f'Features shape: {X_features.shape}')

    # Handle missing values and scale
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_features = imputer.fit_transform(X_features)

    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)

    print('Loading splits...')
    train_df = pd.read_csv('new_objective2_preprocessed/metadata/train_split.csv')
    test_df = pd.read_csv('new_objective2_preprocessed/metadata/test_split.csv')

    train_indices = []
    test_indices = []

    for idx, row in features_df.iterrows():
        if row['image_path'] in train_df['image_path'].values:
            train_indices.append(idx)
        elif row['image_path'] in test_df['image_path'].values:
            test_indices.append(idx)

    X_train = X_features[train_indices]
    X_test = X_features[test_indices]
    y_train = y_features[train_indices]
    y_test = y_features[test_indices]

    print(f'Train: {X_train.shape}, Test: {X_test.shape}')

    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f'Balanced: {X_train_balanced.shape}')

    # Create models
    models = [
        ('lr', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)),
        ('svm', SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ]

    print('Training models...')
    for name, model in models:
        model.fit(X_train_balanced, y_train_balanced)
        print(f'Trained {name}')

    # Create soft voting
    voting_soft = VotingClassifier(estimators=models, voting='soft')
    voting_soft.fit(X_train_balanced, y_train_balanced)

    # Test
    y_pred = voting_soft.predict(X_test)
    y_pred_proba = voting_soft.predict_proba(X_test)

    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.3f}, F1: {f1:.3f}')
    print(f'Probabilities shape: {y_pred_proba.shape}')
    print(f'Sample probabilities: {y_pred_proba[:3]}')

    # Save models
    joblib.dump(voting_soft, 'new_objective2_ensemble_results/models/voting_soft_model.pkl')
    joblib.dump(scaler, 'new_objective2_ensemble_results/models/voting_soft_scaler.pkl')
    joblib.dump(imputer, 'new_objective2_ensemble_results/models/voting_soft_imputer.pkl')

    print('Models saved successfully!')
    return accuracy

if __name__ == "__main__":
    create_proper_model()

