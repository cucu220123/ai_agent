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
    max_features = config.get('max_features', 3000)
    ngram_max = config.get('ngram_max', 1)
    max_iter = config.get('max_iter', 500)
    C = config.get('C', 1.0)
    random_state = config.get('random_state', 42)
    
    # Build pipeline components
    text_preprocessor = FunctionTransformer(
        lambda x: x.fillna('').astype(str),
        validate=False
    )
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max)
    )
    
    classifier = LogisticRegression(
        max_iter=max_iter,
        C=C,
        random_state=random_state,
        solver='liblinear'
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('text_preprocessor', text_preprocessor),
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    # Fit pipeline
    X_train = train_df['text']
    y_train = train_df[target_col]
    pipeline.fit(X_train, y_train)
    
    return pipeline

def predict(model, test_df):
    # Predict using the fitted pipeline
    predictions = model.predict(test_df['text'])
    
    # Return as DataFrame with required column name
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
        'rationale': 'Selected due to its ability to handle text data through TF-IDF vectorization and its effectiveness in binary classification tasks.',
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
            'failure_a054aaacecab_algorithm_tfidf_logistic_regression__tfidf_bigram_v1',
            'failure_a054aaacecab_algorithm_tfidf_logistic_regression__tfidf_unigram_v3'
        ]
    }