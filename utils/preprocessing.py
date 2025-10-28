import numpy as np

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


#Testing

if __name__ == '__main__':
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Original: \n", X)
    print("Mean:", scaler.mean_)
    print("Scale:", scaler.scale_)
    print("Transformed:\n", X_scaled)
    print("Inverse transform:\n", scaler.inverse_transform(X_scaled))