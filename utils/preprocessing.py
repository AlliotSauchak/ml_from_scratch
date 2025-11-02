import numpy as np
import pandas as pd
from collections import Counter

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        return X_scaled * self.scale_ + self.mean_





class SimpleImputer:
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = {}
        self.feature_names_ = None

    def fit(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_ = X.columns.tolist()
        self.statistics_ = {}

        for col in X.columns:
            col_data = X[col].dropna()

            if len(col_data) == 0:
                self.statistics_[col] = self.fill_value if self.fill_value is not None else 0
                continue

            if self.strategy == 'mean':
                if pd.api.types.is_numeric_dtype(X[col]):
                    self.statistics_[col] = col_data.mean()
                else:
                    raise ValueError(f"Cannot use mean strategy on non-numeric column '{col}'")

            elif self.strategy == 'median':
                if pd.api.types.is_numeric_dtype(X[col]):
                    self.statistics_[col] = col_data.median()
                else:
                    raise ValueError(f"Cannot use median strategy on non-numeric column '{col}'")

            elif self.strategy == 'mode':
                most_common = Counter(col_data).most_common(1)
                self.statistics_[col] = most_common[0][0] if most_common else self.fill_value

            elif self.strategy == 'constant':
                self.statistics_[col] = self.fill_value if self.fill_value is not None else 0

            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        return self

    def transform(self, X):
        if not self.statistics_:
            raise ValueError("Imputer must be fitted before transforming data")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_imputed = X.copy()

        for col in X.columns:
            if col in self.statistics_:
                missing_mask = X[col].isna()
                X_imputed.loc[missing_mask, col] = self.statistics_[col]

        return X_imputed

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_statistics(self):
        return self.statistics_
