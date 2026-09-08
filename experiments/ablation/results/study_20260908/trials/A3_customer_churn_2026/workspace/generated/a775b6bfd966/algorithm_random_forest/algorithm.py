import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.exceptions import NotFittedError

def train(train_df, target_col, config=None):
    if config is None:
        config = {}
    
    random_state = config.get('random_state', 42)
    
    # Identify feature columns
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Separate features and target
    X = train_df[feature_cols]
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
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
            min_samples_leaf=1
        ))
    ])
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    return pipeline


def predict(model, test_df):
    # Predict using the fitted pipeline
    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)[:, 1]
    
    # Return DataFrame with exact column names
    return pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })


def predict_proba(model, test_df):
    # Return positive class probabilities
    return model.predict_proba(test_df)[:, 1]


def evaluate(model, test_df, target_col):
    # Split features and target
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, average='binary')
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def metadata():
    return {
        'algorithm': 'Random Forest',
        'rationale': 'Selected based on historical performance and suitability for tabular data with mixed types.',
        'evidence_ids': [
            "9364e416a38a:algorithm_random_forest__llm_proposed:v1",
            "6bde3f2c39b8:algorithm_random_forest__llm_proposed:v4",
            "6bde3f2c39b8:algorithm_random_forest__llm_proposed:v3",
            "6bde3f2c39b8:algorithm_random_forest__llm_proposed:v2",
            "6bde3f2c39b8:algorithm_random_forest__llm_proposed:v1",
            "53ac387cea5c",
            "53ac387cea5c:algorithm_gradient_boosting__llm_proposed:v1",
            "9364e416a38a:algorithm_gradient_boosting__llm_proposed:v1",
            "source_measured_experiment_693b971c"
        ]
    }