import numpy as np


class BinaryNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_priors_ = {}
        self.feature_probs_ = {}

    def fit(self, X, y):
        n, d = X.shape
        self.classes_ = np.unique(y)

        for c in self.classes_:
            X_c = X[y == c]
            n_c = X_c.shape[0]

            self.class_priors_[c] = n_c / n
            self.feature_probs_[c] = (X_c.sum(axis=0) + self.alpha) / (n_c + 2 * self.alpha)

        return self

    def predict_log_proba(self, X):
        log_probs = []

        for x in X:
            class_log_probs = []

            for c in self.classes_:
                log_prob = np.log(self.class_priors_[c])
                p1 = self.feature_probs_[c]
                p0 = 1 - p1
                log_prob += np.sum(x * np.log(p1) + (1 - x) * np.log(p0))

                class_log_probs.append(log_prob)

            log_probs.append(class_log_probs)

        return np.array(log_probs)

    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        predictions = self.classes_[np.argmax(log_probs, axis=1)]
        return predictions


# Testing

if __name__ == '__main__':
    from utils.model_selection import train_test_split
    from utils.metrics import accuracy_score, confusion_matrix
    from sklearn import datasets

    X, y = datasets.make_classification(
        n_samples=500, n_features=10, n_redundant=0, n_informative=8,
        n_clusters_per_class=1, random_state=42
    )

    X_binary = (X > X.mean(axis=0)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y, test_size=0.2, random_state=42
    )

    model = BinaryNaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"Dataset: {X_binary.shape[0]} samples, {X_binary.shape[1]} binary features")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")