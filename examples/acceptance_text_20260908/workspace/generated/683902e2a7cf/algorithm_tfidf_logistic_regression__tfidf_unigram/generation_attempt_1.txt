import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import f1_score, accuracy_score


def train(train_df, target_col, config=None):
    """
    Train a text classification pipeline using TF-IDF and Logistic Regression.
    
    Parameters:
        train_df (pd.DataFrame): Training data with text and target columns.
        target_col (str): Name of the target column.
        config (dict): Configuration dictionary with hyperparameters.
        
    Returns:
        sklearn.pipeline.Pipeline: Fitted pipeline with preprocessing and estimator.
    """
    if config is None:
        config = {}
    
    # Extract text column name
    text_col = 'text'
    
    # Prepare text data
    def text_preprocessor(df):
        return df[text_col].fillna('').astype(str)
    
    # Create pipeline components
    text_transformer = FunctionTransformer(text_preprocessor, validate=False)
    
    # Configure TfidfVectorizer
    max_features = config.get('max_features', 3000)
    ngram_max = config.get('ngram_max', 1)
    ngram_range = (1, ngram_max)
    
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )
    
    # Configure LogisticRegression
    C = config.get('C', 1.0)
    max_iter = config.get('max_iter', 100)
    random_state = config.get('random_state', 42)
    
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state
    )
    
    # Build pipeline
    pipeline = Pipeline([
        ('text', text_transformer),
        ('tfidf', tfidf),
        ('classifier', clf)
    ])
    
    # Fit pipeline
    X_train = train_df[[text_col]]
    y_train = train_df[target_col]
    pipeline.fit(X_train, y_train)
    
    return pipeline


def predict(model, test_df):
    """
    Make predictions using the trained model.
    
    Parameters:
        model (sklearn.pipeline.Pipeline): Trained pipeline.
        test_df (pd.DataFrame): Test data with text column only.
        
    Returns:
        pd.DataFrame: Predictions with column 'prediction'.
    """
    # Ensure we only pass the text column
    predictions = model.predict(test_df)
    
    # Return as DataFrame with required column name
    return pd.DataFrame({'prediction': predictions})


def evaluate(model, test_df, target_col):
    """
    Evaluate the model on test data.
    
    Parameters:
        model (sklearn.pipeline.Pipeline): Trained pipeline.
        test_df (pd.DataFrame): Test data with text and target columns.
        target_col (str): Name of the target column.
        
    Returns:
        dict: Evaluation metrics (f1, accuracy).
    """
    # Split features and target
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    
    return {
        'f1': f1,
        'accuracy': acc
    }


def metadata():
    """
    Return metadata about the algorithm.
    
    Returns:
        dict: Metadata including algorithm name, rationale, and evidence IDs.
    """
    return {
        'algorithm': 'TF-IDF Logistic Regression',
        'rationale': 'This configuration uses TF-IDF vectorization followed by Logistic Regression for text classification. It handles missing values by filling with empty strings and manages unseen vocabulary through the vectorizer.',
        'evidence_ids': [
            "a054aaacecab",
            "a054aaacecab:algorithm_tfidf_logistic_regression__tfidf_bigram:v1",
            "a054aaacecab:algorithm_tfidf_logistic_regression__tfidf_bigram:v2",
            "a054aaacecab:algorithm_tfidf_logistic_regression__tfidf_unigram:v1",
            "a054aaacecab:algorithm_tfidf_logistic_regression__tfidf_unigram:v2",
            "a054aaacecab:algorithm_tfidf_logistic_regression__tfidf_unigram:v3"
        ]
    }