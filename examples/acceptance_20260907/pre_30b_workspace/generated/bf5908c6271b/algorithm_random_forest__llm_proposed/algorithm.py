import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def train(train_df, target_col, config=None):
    """
    Train a Random Forest model for customer churn prediction.

    Parameters:
    - train_df: pandas DataFrame containing the training data.
    - target_col: string, the name of the target column.
    - config: dictionary, optional configuration parameters.

    Returns:
    - fitted sklearn Pipeline containing preprocessing and the estimator.
    """
    # Load configuration
    config = config or {}
    random_state = config.get('random_state', 42)

    # Split the data into features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Preprocessing pipeline
    preprocessor = Pipeline(steps=[
        ('num_imputer', SimpleImputer(strategy='median')),
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Random Forest classifier
    estimator = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])

    # Fit the pipeline
    pipeline.fit(X, y)

    return pipeline

def predict(model, test_df):
    """
    Predict churn probabilities for test data.

    Parameters:
    - model: fitted sklearn Pipeline containing preprocessing and the estimator.
    - test_df: pandas DataFrame containing the test data.

    Returns:
    - pandas DataFrame with columns ['prediction', 'probability'].
    """
    # Ensure the test DataFrame has the same columns as the training data
    if not set(test_df.columns) == set(model.named_steps['preprocessor'].get_feature_names_out()):
        raise ValueError("Test DataFrame must have the same columns as the training data.")

    # Make predictions
    predictions = model.predict_proba(test_df)[:, 1]
    probabilities = pd.DataFrame(predictions, columns=['probability'])

    # Add the prediction column
    test_df['prediction'] = model.predict(test_df)

    # Concatenate predictions and probabilities
    result = pd.concat([test_df[['prediction']], probabilities], axis=1)

    return result

def evaluate(model, test_df, target_col):
    """
    Evaluate the model's performance on the test data.

    Parameters:
    - model: fitted sklearn Pipeline containing preprocessing and the estimator.
    - test_df: pandas DataFrame containing the test data.
    - target_col: string, the name of the target column.

    Returns:
    - dictionary with evaluation metrics: roc_auc, f1, precision, recall.
    """
    # Ensure the test DataFrame has the same columns as the training data
    if not set(test_df.columns) == set(model.named_steps['preprocessor'].get_feature_names_out()):
        raise ValueError("Test DataFrame must have the same columns as the training data.")

    # Split the test data into features and target
    X = test_df.drop(columns=[target_col])
    y = test_df[target_col]

    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Calculate metrics
    roc_auc = roc_auc_score(y, y_prob)
    f1 = f1_score(y, y_pred, average='weighted')
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')

    return {
        'roc_auc': roc_auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def metadata():
    """
    Return metadata about the algorithm.

    Returns:
    - dictionary with algorithm details, rationale, and evidence IDs.
    """
    return {
        'algorithm': 'Random Forest',
        'rationale': 'Random Forest is effective in handling mixed data types and can capture complex interactions between features.',
        'evidence_ids': [
            'source_reference_preprocessing_56f7c1fe',
            'source_business_material_1c1a6f4d',
            'preprocessingstrategy_build_preprocessor'
        ]
    }

def predict_proba(model, test_df):
    """
    Predict probabilities for test data.

    Parameters:
    - model: fitted sklearn Pipeline containing preprocessing and the estimator.
    - test_df: pandas DataFrame containing the test data.

    Returns:
    - 1D numpy array of positive-class probabilities.
    """
    # Ensure the test DataFrame has the same columns as the training data
    if not set(test_df.columns) == set(model.named_steps['preprocessor'].get_feature_names_out()):
        raise ValueError("Test DataFrame must have the same columns as the training data.")

    # Make predictions
    return model.predict_proba(test_df)[:, 1]