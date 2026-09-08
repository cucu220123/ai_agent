import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

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
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
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
        'algorithm': 'Logistic Regression',
        'rationale': '适用于线性关系的简单模型，易于解释。',
        'evidence_ids': ['current_user_requirement']
    }