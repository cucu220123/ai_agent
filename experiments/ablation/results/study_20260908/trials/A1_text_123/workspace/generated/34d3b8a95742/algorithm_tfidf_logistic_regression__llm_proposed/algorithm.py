import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score

def train(train_df, target_col, config=None):
    if config is None:
        config = {}
    
    # Extract hyperparameters from config
    max_features = config.get('max_features', 5000)
    ngram_range = tuple(config.get('ngram_range', [1, 2]))
    max_iter = config.get('max_iter', 100)
    C = config.get('C', 1.0)
    random_state = config.get('random_state', 42)
    
    # Create text processing pipeline
    text_pipeline = Pipeline([
        ('text_selector', FunctionTransformer(lambda x: x.fillna('').astype(str), validate=False)),
        ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ('classifier', LogisticRegression(C=C, max_iter=max_iter, random_state=random_state))
    ])
    
    # Fit the pipeline
    X_train = train_df['text']
    y_train = train_df[target_col]
    text_pipeline.fit(X_train, y_train)
    
    return text_pipeline

def predict(model, test_df):
    # Predict using the fitted pipeline
    predictions = model.predict(test_df['text'])
    
    # Return DataFrame with required column name
    return pd.DataFrame({'prediction': predictions})

def evaluate(model, test_df, target_col):
    # Split features and target
    X_test = test_df['text']
    y_test = test_df[target_col]
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def metadata():
    return {
        'algorithm': 'TF-IDF Logistic Regression',
        'rationale': 'TF-IDF vectorization followed by logistic regression is a standard approach for text classification tasks. It effectively captures term importance and enables linear separation of classes.',
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
            'a054aaacecab:algorithm_tfidf_logistic_regression__llm_proposed:v1'
        ]
    }