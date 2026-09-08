import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column].fillna('').astype(str)


def train(train_df, target_col, config=None):
    if config is None:
        config = {}
    
    # Extract parameters from config
    max_features = config.get('max_features', 5000)
    ngram_range = tuple(config.get('ngram_range', [1, 2]))
    max_iter = config.get('max_iter', 100)
    C = config.get('C', 1.0)
    random_state = config.get('random_state', 42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('text', TextSelector('text')),
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )),
        ('classifier', LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state
        ))
    ])
    
    # Fit the pipeline
    X = train_df[['text']]
    y = train_df[target_col]
    pipeline.fit(X, y)
    
    return pipeline


def predict(model, test_df):
    # Ensure we only pass the text column
    predictions = model.predict(test_df[['text']])
    return pd.DataFrame({'prediction': predictions})


def evaluate(model, test_df, target_col):
    # Split features and target
    X = test_df[['text']]
    y_true = test_df[target_col]
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'f1': f1,
        'accuracy': accuracy
    }


def metadata():
    return {
        'algorithm': 'TF-IDF Logistic Regression',
        'rationale': 'This configuration uses TF-IDF vectorization followed by Logistic Regression, which is effective for text classification tasks. It handles missing values by filling with empty strings and manages unseen vocabulary through the vectorizer.',
        'evidence_ids': [
            'a054aaacecab:algorithm_tfidf_logistic_regression__tfidf_bigram:v2',
            'a054aaacecab:algorithm_tfidf_logistic_regression__tfidf_bigram:v1',
            'a054aaacecab',
            'a054aaacecab:algorithm_tfidf_logistic_regression__tfidf_unigram:v3',
            'a054aaacecab:algorithm_tfidf_logistic_regression__tfidf_unigram:v2',
            'a054aaacecab:algorithm_tfidf_logistic_regression__tfidf_unigram:v1',
            'a054aaacecab:algorithm_tfidf_logistic_regression__llm_proposed:v4',
            'a054aaacecab:algorithm_tfidf_logistic_regression__llm_proposed:v3',
            'a054aaacecab:algorithm_tfidf_logistic_regression__llm_proposed:v2',
            'a054aaacecab:algorithm_tfidf_logistic_regression__llm_proposed:v1',
            'failure_a054aaacecab_algorithm_tfidf_logistic_regression__tfidf_unigram_v3',
            'repair_a054aaacecab_algorithm_tfidf_logistic_regression__tfidf_unigram_v3'
        ]
    }