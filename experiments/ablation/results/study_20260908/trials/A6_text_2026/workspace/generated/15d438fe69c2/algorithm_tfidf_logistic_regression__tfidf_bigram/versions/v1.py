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
    ngram_max = config.get('ngram_max', 2)
    max_iter = config.get('max_iter', 500)
    C = config.get('C', 1.0)
    random_state = config.get('random_state', 42)
    
    # Build ngram_range from ngram_max
    ngram_range = (1, ngram_max)
    
    # Create pipeline components
    text_preprocessor = FunctionTransformer(
        lambda x: x.fillna('').astype(str),
        validate=False
    )
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )
    
    classifier = LogisticRegression(
        max_iter=max_iter,
        C=C,
        random_state=random_state,
        solver='liblinear'
    )
    
    # Construct pipeline
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
        'rationale': 'Meets the requirement for text classification with TF-IDF and Logistic Regression, supports comparison of different configurations.',
        'evidence_ids': ['current_user_requirement']
    }