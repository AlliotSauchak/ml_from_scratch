import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_idx = np.argsort(distances)[:self.k]
            values, counts = np.unique(self.y_train[k_idx], return_counts=True)
            predictions.append(values[np.argmax(counts)])
        return np.array(predictions)


if __name__ == "__main__":
    from sklearn import datasets
    from utils.model_selection import train_test_split
    from utils.preprocessing import StandardScaler
    from utils.metrics import accuracy_score, precision_score, recall_score

    X, y = datasets.load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNN(k=5)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
