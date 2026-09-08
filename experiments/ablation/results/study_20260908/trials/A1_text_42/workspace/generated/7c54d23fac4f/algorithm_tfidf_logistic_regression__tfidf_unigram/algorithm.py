import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import NotFittedError

def train(train_df, target_col, config=None):
    if config is None:
        config = {}
    
    # Extract hyperparameters from config
    max_features = config.get('max_features', 3000)
    ngram_max = config.get('ngram_max', 1)
    max_iter = config.get('max_iter', 500)
    C = config.get('C', 1.0)
    random_state = config.get('random_state', 42)
    class_weight = config.get('class_weight', None)
    
    # Build pipeline components
    text_preprocessor = FunctionTransformer(
        lambda df: df.iloc[:, 0].fillna('').astype(str) if isinstance(df, pd.DataFrame) else pd.Series(df).fillna('').astype(str),
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
        solver='liblinear',
        class_weight=class_weight
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('text_preprocessor', text_preprocessor),
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    # Fit pipeline
    X_train = train_df.iloc[:, 0]  # Get first column as text
    y_train = train_df[target_col]
    pipeline.fit(X_train, y_train)
    
    return pipeline

def predict(model, test_df):
    # Predict using the fitted pipeline
    predictions = model.predict(test_df.iloc[:, 0])
    
    # Return DataFrame with required column name
    return pd.DataFrame({'prediction': predictions})

def evaluate(model, test_df, target_col):
    # Split features and target
    X_test = test_df.iloc[:, 0]
    y_test = test_df[target_col]
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Return metrics as dictionary
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def metadata():
    return {
        'algorithm': 'TF-IDF Logistic Regression',
        'rationale': 'Selected due to its simplicity, interpretability, and ability to handle text data effectively through TF-IDF vectorization.',
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
