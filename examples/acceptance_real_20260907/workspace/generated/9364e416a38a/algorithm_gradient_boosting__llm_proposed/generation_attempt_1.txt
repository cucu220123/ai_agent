import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score


def train(train_df, target_col, config=None):
    """
    Train a binary classification model using Gradient Boosting.
    
    Parameters:
        train_df (pd.DataFrame): Training data including features and target.
        target_col (str): Name of the target column.
        config (dict): Configuration dictionary, supports 'random_state'.
        
    Returns:
        sklearn Pipeline: Fitted preprocessing and estimator pipeline.
    """
    if config is None:
        config = {}
    
    random_state = config.get('random_state', 42)
    
    # Separate features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
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
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state
        ))
    ])
    
    # Fit the model
    model.fit(X, y)
    
    return model


def predict(model, test_df):
    """
    Make predictions on test data.
    
    Parameters:
        model (sklearn Pipeline): Fitted model.
        test_df (pd.DataFrame): Test data without target column.
        
    Returns:
        pd.DataFrame: Predictions with 'prediction' and 'probability' columns.
    """
    # Get predictions and probabilities
    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)[:, 1]
    
    # Return as DataFrame
    return pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })


def predict_proba(model, test_df):
    """
    Get positive class probabilities from the model.
    
    Parameters:
        model (sklearn Pipeline): Fitted model.
        test_df (pd.DataFrame): Test data without target column.
        
    Returns:
        np.ndarray: Array of positive class probabilities.
    """
    return model.predict_proba(test_df)[:, 1]


def evaluate(model, test_df, target_col):
    """
    Evaluate model performance on test data.
    
    Parameters:
        model (sklearn Pipeline): Fitted model.
        test_df (pd.DataFrame): Labeled test data.
        target_col (str): Name of the target column.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
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
    """
    Return metadata about the algorithm.
    
    Returns:
        dict: Metadata including algorithm name, rationale, and evidence IDs.
    """
    return {
        'algorithm': 'Gradient Boosting',
        'rationale': 'Strong performance in handling class imbalance and high ROC-AUC scores.',
        'evidence_ids': [
            '6bde3f2c39b8:algorithm_gradient_boosting__llm_proposed:v1',
            '6bde3f2c39b8:algorithm_random_forest__llm_proposed:v4',
            '6bde3f2c39b8:algorithm_random_forest__llm_proposed:v3',
            '6bde3f2c39b8:algorithm_random_forest__llm_proposed:v2'
        ]
    }