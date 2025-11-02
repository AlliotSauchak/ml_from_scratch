import numpy as np


class Ridge:
    def __init__(self, lr=0.01, epochs=1000, reg_coef=0.01):
        self.lr = lr
        self.epochs = epochs
        self.reg_coef = reg_coef

        self.weights_ = None
        self.bias_ = None
        self.loss_history_ = []

    def fit(self, X, y):
        n, d = X.shape
        self.weights_ = np.random.randn(X.shape[1]) * 0.01
        self.bias_ = 0

        for epoch in range(self.epochs):
            h = X @ self.weights_ + self.bias_

            loss = np.mean((y - h) ** 2) + self.reg_coef * np.sum(self.weights_ ** 2)
            self.loss_history_.append(loss)

            grad_w = (-2 * X.T @ (y - h)) / n + 2 * self.reg_coef*self.weights_
            grad_b = np.sum((h - y)) / n

            self.weights_ -= self.lr * grad_w
            self.bias_ -= self.lr * grad_b

        return self

    def predict(self, X):
        prediction = X @ self.weights_ + self.bias_
        return prediction


if __name__ == '__main__':
    from utils.model_selection import train_test_split
    from utils.preprocessing import StandardScaler
    from utils.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    from sklearn import datasets

    X, y = datasets.make_regression(
        n_samples=200, n_features=1, noise=30, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge()
    model.fit(X_train_scaled, y_train)

    y_test_pred = model.predict(X_test_scaled)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"MSE: {mean_squared_error(y_test, y_test_pred)}")
    print(f"MAE: {mean_absolute_error(y_test, y_test_pred)}")
    print(f"RMSE: {root_mean_squared_error(y_test, y_test_pred)}")
    print(f"R2: {r2_score(y_test, y_test_pred)}")

    y_predictions = model.predict(X)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, label='Text Indices', alpha=0.6)
    plt.plot(X_test, y_test_pred, label='Predictions', alpha=0.6, c='red')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()


    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_history_, label='Training Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()