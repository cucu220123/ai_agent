import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score


def train(train_df, target_col, config=None):
    """Train a Gradient Boosting model with preprocessing."""
    if config is None:
        config = {}
    
    random_state = config.get('random_state', 42)
    
    # Identify feature columns
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Separate features and target
    X = train_df[feature_cols]
    y = train_df[target_col]
    
    # Define numeric and categorical columns
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
    
    # Model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            max_depth=config.get('max_depth', 3),
            random_state=random_state
        ))
    ])
    
    # Fit the model
    model.fit(X, y)
    
    return model


def predict(model, test_df):
    """Predict using the trained model."""
    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)[:, 1]
    
    return pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })


def predict_proba(model, test_df):
    """Return positive class probabilities."""
    return model.predict_proba(test_df)[:, 1]


def evaluate(model, test_df, target_col):
    """Evaluate model performance."""
    # Split features and target
    X = test_df.drop(columns=[target_col])
    y_true = test_df[target_col]
    
    # Predictions
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
    """Return model metadata."""
    return {
        'algorithm': 'Gradient Boosting',
        'rationale': 'Gradient Boosting is suitable for imbalanced datasets and can handle both numerical and categorical features.',
        'evidence_ids': [
            'source_business_material_1c1a6f4d',
            'preprocessing_3b20f945c980b1ea',
            'preprocessing_56a78a1499364981',
            'preprocessing_8817df495c5b2800',
            'preprocessing_8d29697f8be324cb',
            'preprocessing_e30ccf02b0837f02',
            'task_binary_classification'
        ]
    }