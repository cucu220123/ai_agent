import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, f1_score

def train(train_df, target_col, config=None):
    """Train a TF-IDF Logistic Regression pipeline for text classification."""
    if config is None:
        config = {}
    
    random_state = config.get('random_state', 42)
    
    # Create a function transformer to handle text column and missing values
    text_transformer = FunctionTransformer(
        lambda x: x.fillna('').astype(str),
        validate=False
    )
    
    # Build the pipeline
    pipeline = Pipeline([
        ('text', text_transformer),
        ('tfidf', TfidfVectorizer(
            max_features=config.get('max_features', 3000),
            ngram_range=tuple(config.get('ngram_range', [1, 2]))
        )),
        ('classifier', LogisticRegression(
            random_state=random_state,
            max_iter=config.get('max_iter', 100),
            C=config.get('C', 1.0)
        ))
    ])
    
    # Fit the pipeline
    X_train = train_df['text']
    y_train = train_df[target_col]
    pipeline.fit(X_train, y_train)
    
    return pipeline

def predict(model, test_df):
    """Predict using the trained model."""
    predictions = model.predict(test_df['text'])
    return pd.DataFrame({'prediction': predictions})

def evaluate(model, test_df, target_col):
    """Evaluate the model on test data."""
    X_test = test_df['text']
    y_test = test_df[target_col]
    
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def metadata():
    """Return metadata about the algorithm."""
    return {
        'algorithm': 'TF-IDF Logistic Regression',
        'rationale': 'Selected due to its simplicity, interpretability, and suitability for text classification tasks.',
        'evidence_ids': [
            'source_text_material_84554427',
            'source_reference_preprocessing_56f7c1fe',
            'algorithm_tfidf_logistic_regression'
        ]
    }
