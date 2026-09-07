import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, MostFrequentImputer
import numpy as np

def train(train_df, target_col, config=None):
    # Extract features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Preprocessing steps with imputation
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', MostFrequentImputer()),
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    # Model configuration
    random_state = config.get('random_state', 42) if config else 42
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    return pipeline

def predict(model, test_df):
    # Predict using the fitted pipeline
    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)[:, 1]  # Probability of positive class
    
    # Return DataFrame with required column names
    return pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })

def predict_proba(model, test_df):
    # Return probability of positive class
    return model.predict_proba(test_df)[:, 1]

def evaluate(model, test_df, target_col):
    # Separate features and target
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    f1 = f1_score(y_test, y_pred, average='binary')
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_prob)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

def metadata():
    return {
        'algorithm': 'Random Forest',
        'rationale': 'Random Forest is used for binary classification due to its robustness, ability to handle mixed data types, and built-in feature importance.',
        'evidence_ids': ['rf_binary_classification']
    }