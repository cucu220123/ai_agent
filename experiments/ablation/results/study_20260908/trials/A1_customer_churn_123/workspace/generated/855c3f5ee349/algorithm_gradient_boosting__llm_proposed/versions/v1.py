import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

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
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
    )
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            max_depth=config.get('max_depth', 3),
            random_state=random_state
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
        'algorithm': 'Gradient Boosting',
        'rationale': '梯度提升算法在处理类别不平衡数据集时表现出色，并且在之前的历史运行中取得了较高的ROC-AUC分数。',
        'evidence_ids': [
            '9364e416a38a:algorithm_gradient_boosting__llm_proposed:v1',
            '53ac387cea5c',
            '53ac387cea5c:algorithm_gradient_boosting__llm_proposed:v1',
            '9364e416a38a:algorithm_random_forest__llm_proposed:v1',
            'failure_9364e416a38a_algorithm_random_forest__llm_proposed_v1',
            'failure_0f7edacd5655_algorithm_logistic_regression__llm_proposed_v1',
            'repair_53ac387cea5c_algorithm_gradient_boosting__llm_proposed_v1',
            'source_business_material_1c1a6f4d'
        ]
    }