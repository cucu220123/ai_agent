import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.utils import resample

def build_preprocessor(features, config={}):
    num_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = features.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_features),
            ('cat', [('impute', SimpleImputer(strategy='most_frequent')), 
                     ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))], cat_features)
        ]
    )
    return preprocessor

def train(train_df, target_col, config=None):
    if config is None:
        config = {}
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # Handling class imbalance using Random Over-Sampling
    X_resampled, y_resampled = resample(X_train[y_train == 1], y_train[y_train == 1], replace=True, n_samples=X_train[y_train == 0].shape[0], random_state=config.get('random_state', 42))
    X_resampled = pd.concat([X_resampled, X_train[y_train == 0]])
    y_resampled = pd.concat([y_resampled, y_train[y_train == 0]])

    preprocessor = build_preprocessor(X_resampled, config)
    X_resampled_processed = preprocessor.fit_transform(X_resampled)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=config.get('random_state', 42))
    model.fit(X_resampled_processed, y_resampled)
    
    return model, preprocessor

def predict(model, test_df):
    X_test = test_df
    X_test_processed = model.preprocessor.transform(X_test)
    predictions = model.predict(X_test_processed)
    probabilities = model.predict_proba(X_test_processed)[:, 1]
    
    result_df = pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })
    return result_df

def predict_proba(model, test_df):
    X_test = test_df
    X_test_processed = model.preprocessor.transform(X_test)
    probabilities = model.predict_proba(X_test_processed)[:, 1]
    return probabilities

def evaluate(model, test_df, target_col):
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    X_test_processed = model.preprocessor.transform(X_test)
    
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
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
        'rationale': 'Robust to overfitting and can handle mixed data types effectively.',
        'evidence_ids': ['source_reference_preprocessing_7238a567', 'source_business_material_9b240e8c', 'preprocessingstrategy_build_preprocessor', 'feature_region']
    }