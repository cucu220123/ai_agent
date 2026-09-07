import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

def train(train_df, target_col, config=None):
    if config is None:
        config = {}
    random_state = config.get('random_state', 42)
    
    # Separate features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            C=config.get('C', 1.0),
            max_iter=config.get('max_iter', 100),
            class_weight=config.get('class_weight', 'balanced'),
            random_state=random_state
        ))
    ])
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    return pipeline

def predict_broken(model, test_df):
    # Predict classes and probabilities
    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)[:, 1]
    
    # Return as DataFrame with required column names
    return pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })

def predict_proba(model, test_df):
    # Return positive class probabilities
    return model.predict_proba(test_df)[:, 1]

def evaluate(model, test_df, target_col):
    # Split features and target
    X = test_df.drop(columns=[target_col])
    y_true = test_df[target_col]
    
    # Get predictions and probabilities
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Compute metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def metadata():
    return {
        'algorithm': 'Logistic Regression',
        'rationale': 'Historical run shows high ROC-AUC score and acceptable F1, precision, and recall metrics.',
        'evidence_ids': [
            '6bde3f2c39b8',
            '9364e416a38a',
            '6bde3f2c39b8:algorithm_gradient_boosting__llm_proposed:v1',
            '9364e416a38a:algorithm_gradient_boosting__llm_proposed:v1'
        ]
    }