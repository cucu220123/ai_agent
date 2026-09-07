import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, MostFrequentImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
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
        ('imputer', MostFrequentImputer(strategy='most_frequent')),
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
            max_depth=3,
            random_state=random_state,
            min_samples_leaf=1
        ))
    ])
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    return pipeline

def predict(model, test_df):
    # Get predictions and probabilities
    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)[:, 1]
    
    # Return as DataFrame with exact column names
    return pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })

def predict_proba(model, test_df):
    # Return positive class probability
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
        'algorithm': 'Random Forest',
        'rationale': 'Selected for its ability to handle mixed data types, robustness to outliers, and good performance on tabular data with class imbalance.',
        'evidence_ids': [
            '6bde3f2c39b8:algorithm_random_forest__llm_proposed:v4',
            '6bde3f2c39b8:algorithm_random_forest__llm_proposed:v3',
            '6bde3f2c39b8:algorithm_random_forest__llm_proposed:v2',
            '6bde3f2c39b8:algorithm_random_forest__llm_proposed:v1',
            '6bde3f2c39b8:algorithm_gradient_boosting__llm_proposed:v1'
        ]
    }