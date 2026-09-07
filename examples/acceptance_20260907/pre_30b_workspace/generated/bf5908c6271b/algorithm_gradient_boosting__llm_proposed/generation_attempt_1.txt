import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def train(train_df, target_col, config=None):
    """
    Train a Gradient Boosting classifier on the given training data.

    Parameters:
    - train_df: pandas DataFrame containing the training data.
    - target_col: string, the name of the target column.
    - config: optional dictionary, configuration parameters.

    Returns:
    - fitted sklearn Pipeline containing preprocessing and the estimator.
    """
    # Load configuration
    config = config or {}
    random_state = config.get('random_state', 42)

    # Split the data into features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Define preprocessing steps
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', None)  # No scaling needed for numeric features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the estimator
    estimator = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state)

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])

    # Fit the pipeline
    pipeline.fit(X, y)

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
    # Ensure the test data has the same columns as the training data
    if not set(test_df.columns).issubset(set(model.named_steps['preprocessor'].transformers[0][1].get_feature_names_out())):
        raise ValueError("Test data must have the same columns as the training data.")

    # Make predictions
    predictions = model.predict_proba(test_df)[:, 1]
    df_predictions = pd.DataFrame({'prediction': predictions})

    return df_predictions

def evaluate(model, test_df, target_col):
    """
    Evaluate the model on the given test data.

    Parameters:
    - model: fitted sklearn Pipeline containing preprocessing and the estimator.
    - test_df: pandas DataFrame containing the test data.
    - target_col: string, the name of the target column.

    Returns:
    - dictionary with metrics: roc_auc, f1, precision, recall.
    """
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