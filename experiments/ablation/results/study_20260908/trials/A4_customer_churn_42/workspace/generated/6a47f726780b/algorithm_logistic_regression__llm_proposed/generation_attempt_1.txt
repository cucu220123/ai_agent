import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

def train(train_df, target_col, config=None):
    if config is None:
        config = {}
    random_state = config.get('random_state', 42)
    
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    numeric_features = ['age', 'login_count_30d', 'total_spend', 'complaint_count', 'tenure_months']
    categorical_features = ['region', 'membership_level']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=1.0, max_iter=100, random_state=random_state))
    ])
    
    model.fit(X, y)
    return model

def predict(model, test_df):
    prediction = model.predict(test_df)
    probability = model.predict_proba(test_df)[:, 1]
    
    return pd.DataFrame({
        'prediction': prediction,
        'probability': probability
    })

def predict_proba(model, test_df):
    return model.predict_proba(test_df)[:, 1]

def evaluate(model, test_df, target_col):
    X = test_df.drop(columns=[target_col])
    y = test_df[target_col]
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    roc_auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    f1 = f1_score(y, y_pred, average='binary')
    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')
    
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
        'rationale': 'Commonly used for binary classification tasks with good interpretability and performance.',
        'evidence_ids': [
            '9364e416a38a',
            '0f7edacd5655',
            '2b2a3d8a9bac',
            '6bde3f2c39b8',
            '53ac387cea5c',
            '53ac387cea5c:algorithm_gradient_boosting__llm_proposed:v1',
            '9364e416a38a:algorithm_gradient_boosting__llm_proposed:v1',
            '6bde3f2c39b8:algorithm_random_forest__llm_proposed:v4',
            'source_measured_experiment_693b971c'
        ]
    }