import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def train(train_df, target_col, config=None):
    """Train a churn prediction model with preprocessing."""
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
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
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
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=1.0, max_iter=500, random_state=random_state))
    ])
    
    # Fit the model
    model.fit(X, y)
    
    return model


def predict(model, test_df):
    """Predict churn probability and class labels."""
    # Get predictions and probabilities
    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)[:, 1]
    
    # Return as DataFrame with required column names
    return pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })


def predict_proba(model, test_df):
    """Return positive class probability."""
    return model.predict_proba(test_df)[:, 1]


def evaluate(model, test_df, target_col):
    """Evaluate model performance on test data."""
    # Split features and target
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, average='binary')
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    
    return {
        'roc_auc': roc_auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def metadata():
    """Return model metadata."""
    return {
        'algorithm': 'Logistic Regression',
        'rationale': 'Logistic Regression provides a simple baseline model and is interpretable.',
        'evidence_ids': [
            'source_reference_preprocessing_56f7c1fe',
            'source_business_material_1c1a6f4d',
            'preprocessingstrategy_build_preprocessor'
        ]
    }