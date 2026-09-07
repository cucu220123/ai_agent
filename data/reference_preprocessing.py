"""Reference capability: mixed numeric/categorical tabular preprocessing.

Suitable for binary classification and regression. Numerical missing values
use median imputation, categories use most-frequent imputation and one-hot
encoding with unseen category handling. Requires scikit-learn and numpy.
The supervised target must be excluded before calling this function.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(features, scale=True):
    """Return an unfitted ColumnTransformer; fit on training features only."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(steps), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
