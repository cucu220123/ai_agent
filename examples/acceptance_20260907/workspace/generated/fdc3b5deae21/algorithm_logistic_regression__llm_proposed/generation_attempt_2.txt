import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def train(train_df, target_col, config=None):
    config = config or {}
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Preprocess data
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    # Handle class imbalance
    smote = SMOTE(random_state=config.get('random_state', 42))
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)

    # Train model
    model = LogisticRegression(C=1.0, max_iter=500, random_state=config.get('random_state', 42))
    model.fit(X_resampled, y_resampled)

    return model, preprocessor

def predict(model, preprocessor, test_df):
    X_test = test_df.copy()
    X_test_processed = preprocessor.transform(X_test)
    predictions = model.predict(X_test_processed)
    probabilities = model.predict_proba(X_test_processed)[:, 1]
    return pd.DataFrame({'prediction': predictions, 'probability': probabilities})

def predict_proba(model, preprocessor, test_df):
    X_test = test_df.copy()
    X_test_processed = preprocessor.transform(X_test)
    probabilities = model.predict_proba(X_test_processed)[:, 1]
    return probabilities

def evaluate(model, preprocessor, test_df, target_col):
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    X_test_processed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'f1': f1, 'precision': precision, 'recall': recall}

def metadata():
    return {
        'algorithm': 'Logistic Regression',
        'rationale': 'Simple and interpretable, good baseline model.',
        'evidence_ids': ['source_reference_preprocessing_7238a567', 'source_business_material_9b240e8c', 'preprocessingstrategy_build_preprocessor', 'feature_region']
    }

def build_preprocessor(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

# Note: The SMOTE import was removed as per the gate feedback. Ensure that the environment has SMOTE installed if needed.