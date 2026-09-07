import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def train(train_df, target_col, config=None):
    if config is None:
        config = {}
    
    random_state = config.get('random_state', 42)
    
    # Separate features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ]
    )
    
    # Model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state
        ))
    ])
    
    model.fit(X, y)
    return model

def predict(model, test_df):
    # Predict using the fitted pipeline
    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)[:, 1]
    
    # Return DataFrame with required column names
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
    y = test_df[target_col]
    
    # Get predictions and probabilities
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Compute metrics
    roc_auc = roc_auc_score(y, y_prob)
    f1 = f1_score(y, y_pred, average='binary')
    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')
    
    return {
        'roc_auc': roc_auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def metadata():
    return {
        'algorithm': 'Gradient Boosting',
        'rationale': 'Gradient Boosting can handle class imbalance and provide robust performance with proper tuning.',
        'evidence_ids': [
            'source_reference_preprocessing_56f7c1fe',
            'source_business_material_1c1a6f4d',
            'preprocessingstrategy_build_preprocessor'
        ]
    }