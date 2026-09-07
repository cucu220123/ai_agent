import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

class CoderAgent:
    def __init__(self):
        self.preprocessor = None
        self.model = None

    def train(self, train_df, target_col='churn', config=None):
        if config is None:
            config = {}
        
        # Extract features and target
        X = train_df.drop(columns=[target_col])
        y = train_df[target_col]
        
        # Define preprocessing steps
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X.select_dtypes(include=['object']).columns
        
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())  # Assuming we need scaling for numerical features
        ])
        
        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, num_features),
                ('cat', cat_pipeline, cat_features)
            ]
        )
        
        # Fit the preprocessor on training data
        preprocessor.fit(X)
        
        # Transform the training data
        X_transformed = preprocessor.transform(X)
        
        # Define the model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=config.get('random_state', 42))
        
        # Train the model
        self.model.fit(X_transformed, y)
        
        # Store the preprocessor
        self.preprocessor = preprocessor
        
        return {
            'preprocessor': self.preprocessor,
            'model': self.model
        }

    def predict(self, model, test_df):
        if model is None:
            raise ValueError("Model not found. Please train the model first.")
        
        # Extract features
        X = test_df.drop(columns=['churn'])
        
        # Transform the test data using the same preprocessor
        X_transformed = self.preprocessor.transform(X)
        
        # Make predictions
        predictions = model.predict(X_transformed)
        probabilities = model.predict_proba(X_transformed)[:, 1]
        
        # Create a DataFrame with predictions and probabilities
        result_df = pd.DataFrame({
            'predicted_churn': predictions,
            'positive_class_probability': probabilities
        })
        
        return result_df

    def predict_proba(self, model, test_df):
        if model is None:
            raise ValueError("Model not found. Please train the model first.")
        
        # Extract features
        X = test_df.drop(columns=['churn'])
        
        # Transform the test data using the same preprocessor
        X_transformed = self.preprocessor.transform(X)
        
        # Get probabilities
        probabilities = model.predict_proba(X_transformed)[:, 1]
        
        return probabilities

    def evaluate(self, model, test_df, target_col='churn'):
        if model is None:
            raise ValueError("Model not found. Please train the model first.")
        
        # Extract features and target
        X = test_df.drop(columns=[target_col])
        y = test_df[target_col]
        
        # Transform the test data using the same preprocessor
        X_transformed = self.preprocessor.transform(X)
        
        # Make predictions
        y_pred = model.predict(X_transformed)
        y_prob = model.predict_proba(X_transformed)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y, y_prob)
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        
        return {
            'roc_auc': roc_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def metadata(self):
        return {
            'algorithm': 'Random Forest',
            'rationale': 'Random Forest is effective in handling mixed data types and can capture complex interactions between features.',
            'evidence_ids': [
                'source_reference_preprocessing_56f7c1fe',
                'source_business_material_1c1a6f4d',
                'preprocessingstrategy_build_preprocessor'
            ]
        }