import numpy as np



class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

        self.loss_history_ = []
        self.weights_ = None
        self.bias_ = None

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def fit(self, X, y):
        n, d = X.shape
        self.weights_ = np.random.rand(d) * 0.01
        self.bias_ = 0.0

        for epoch in range(self.epochs):
            z = X @ self.weights_ + self.bias_
            h = self.sigmoid(z)

            grad_w = X.T @ (h - y) / n
            grad_b = np.sum((h - y)) / n

            loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
            self.loss_history_.append(loss)

            self.weights_ -= self.lr * grad_w
            self.bias_ -= self.lr * grad_b

        return self

    def predict(self, X):
        z = X @ self.weights_ + self.bias_
        return np.where(self.sigmoid(z) >= 0.5, 1, 0)


#Testing
if __name__ == "__main__":
    from sklearn import datasets
    from utils.model_selection import train_test_split
    from utils.preprocessing import StandardScaler
    from utils.metrics import accuracy, precision_score, recall_score, f1_score

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    lr = LogisticRegression(epochs=1000)
    lr.fit(X_train, y_train)

    predictions = lr.predict(X_test)

    print('Accuracy:', accuracy(y_test, predictions))
    print('Precision:', precision_score(y_test, predictions))
    print('Recall:', recall_score(y_test, predictions))
    print('f1:', f1_score(y_test, predictions))
