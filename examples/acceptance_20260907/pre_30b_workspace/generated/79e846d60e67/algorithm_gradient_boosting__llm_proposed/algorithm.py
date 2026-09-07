import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def train(train_df, target_col, config=None):
    """
    Train a Gradient Boosting classifier on the given training data.

    Parameters:
    - train_df: pandas DataFrame containing the training data.
    - target_col: string, the name of the target column.
    - config: dictionary, optional configuration parameters.

    Returns:
    - fitted sklearn Pipeline containing preprocessing and the estimator.
    """
    if config is None:
        config = {}

    # Extract features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=config.get('random_state', 42))

    # Build preprocessing pipeline
    preprocessor = Pipeline(steps=[
        ('num_imputer', SimpleImputer(strategy='median')),
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Build the Gradient Boosting classifier
    estimator = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=config.get('random_state', 42))

    # Create the final pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', estimator)
    ])

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    return pipeline

def predict(model, test_df):
    """
    Predict churn probabilities for the given test data.

    Parameters:
    - model: fitted sklearn Pipeline containing preprocessing and the estimator.
    - test_df: pandas DataFrame containing the test data.

    Returns:
    - pandas DataFrame with columns ['prediction', 'probability'].
    """
    # Extract features
    X_test = test_df

    # Predict probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Create the result DataFrame
    result_df = pd.DataFrame({
        'prediction': model.predict(X_test),
        'probability': probabilities
    })

    return result_df

def evaluate(model, test_df, target_col):
    """
    Evaluate the model's performance on the given test data.

    Parameters:
    - model: fitted sklearn Pipeline containing preprocessing and the estimator.
    - test_df: pandas DataFrame containing the test data.
    - target_col: string, the name of the target column.

    Returns:
    - dictionary with metrics: roc_auc, f1, precision, recall.
    """
    # Extract features and target
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Predict probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, probabilities)
    f1 = f1_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))

    return {
        'roc_auc': roc_auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def metadata():
    """
    Return metadata about the algorithm used.

    Returns:
    - dictionary with algorithm, rationale, and evidence_ids.
    """
    return {
        'algorithm': 'Gradient Boosting',
        'rationale': 'Gradient Boosting can handle class imbalance and provide robust performance with proper tuning.',
        'evidence_ids': [
            'source_reference_preprocessing_56f7c1fe',
            'source_business_material_1c1a6f4d',
            'preprocessingstrategy_build_preprocessor'
        ]
    }

def predict_proba(model, test_df):
    """
    Predict probabilities for the given test data.

    Parameters:
    - model: fitted sklearn Pipeline containing preprocessing and the estimator.
    - test_df: pandas DataFrame containing the test data.

    Returns:
    - 1D numpy array of positive-class probabilities.
    """
    return model.predict_proba(test_df)[:, 1]