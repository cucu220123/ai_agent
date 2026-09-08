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
    
    # Extract text column name
    text_col = 'text'
    
    # Build pipeline components
    # Text processing step: select text column, fill missing with empty string, convert to str
    text_selector = FunctionTransformer(
        lambda x: x.fillna('').astype(str),
        validate=False
    )
    
    # TF-IDF vectorizer
    max_features = config.get('max_features', 3000)
    ngram_max = config.get('ngram_max', 1)
    ngram_range = (1, ngram_max)
    
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )
    
    # Classifier
    C = config.get('C', 1.0)
    max_iter = config.get('max_iter', 500)
    random_state = config.get('random_state', 42)
    
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('text', text_selector),
        ('tfidf', tfidf),
        ('classifier', clf)
    ])
    
    # Fit pipeline
    X_train = train_df[text_col]
    y_train = train_df[target_col]
    pipeline.fit(X_train, y_train)
    
    return pipeline

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
        'rationale': 'Selected based on the requirement to use TF-IDF + Logistic Regression for text classification task.',
        'evidence_ids': [
            'dependency_scikit_learn',
            'environment_python_sklearn',
            'preprocessing_5f4db9089bef0887',
            'preprocessing_9929dd86179e6047',
            'preprocessing_a7ba449fb69f797f',
            'preprocessing_f3eff5a94df7aea4',
            'source_text_material_84554427',
            'task_text_classification'
        ]
    }