import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score

def train(train_df, target_col, config=None):
    # Extract features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Define preprocessing steps
    numeric_features = ['age', 'login_count_30d', 'total_spend', 'complaint_count', 'tenure_months']
    categorical_features = ['region', 'membership_level']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Define the classifier
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=config.get('random_state', 42))
    
    # Create and fit the pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    pipe.fit(X, y)
    
    return pipe

def predict(model, test_df):
    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)[:, 1]
    return pd.DataFrame({'prediction': predictions, 'probability': probabilities})

def evaluate(model, test_df, target_col):
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, average='binary')
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'f1': f1, 'precision': precision, 'recall': recall}

def predict_proba(model, test_df):
    return model.predict_proba(test_df)[:, 1]

def metadata():
    return {
        'algorithm': 'Gradient Boosting',
        'rationale': 'Gradient Boosting can handle class imbalance and provide robust performance with proper tuning.',
        'evidence_ids': ['source_reference_preprocessing_56f7c1fe', 'source_business_material_1c1a6f4d', 'preprocessingstrategy_build_preprocessor']
    }