import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def train(train_df, target_col, config=None):
    # Identify numeric and categorical columns
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    # Drop the target column
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Define imputers
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Define encoders
    categorical_encoder = OneHotEncoder(handle_unknown='ignore')
    
    # Define transformers
    transformers = [
        ('num', numeric_imputer, numeric_cols),
        ('cat', categorical_imputer, categorical_cols)
    ]
    
    # Create a ColumnTransformer
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Define the estimator
    estimator = LogisticRegression(C=config.get('C', 1.0), max_iter=config.get('max_iter', 500), class_weight=config.get('class_weight', 'balanced'))
    
    # Create a pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('estimator', estimator)])
    
    # Fit the model
    pipeline.fit(X, y)
    
    return pipeline

def predict(model, test_df):
    # Drop the target column
    X_test = test_df.drop(columns=['churn'])
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create a DataFrame with predictions and probabilities
    result_df = pd.DataFrame({
        'prediction': model.predict(X_test),
        'probability': y_pred_proba
    })
    
    return result_df

def evaluate(model, test_df, target_col):
    # Predict probabilities
    y_pred_proba = model.predict_proba(test_df.drop(columns=[target_col]))[:, 1]
    
    # Predict labels
    y_pred = model.predict(test_df.drop(columns=[target_col]))
    
    # Calculate metrics
    roc_auc = roc_auc_score(test_df[target_col], y_pred_proba)
    f1 = f1_score(test_df[target_col], y_pred)
    precision = precision_score(test_df[target_col], y_pred)
    recall = recall_score(test_df[target_col], y_pred)
    
    return {
        'roc_auc': roc_auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }