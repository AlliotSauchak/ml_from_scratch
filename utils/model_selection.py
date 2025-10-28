import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True):
    X = np.array(X)
    y = np.array(y)

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    test_count = int(n_samples * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test

#Testing

if __name__ == '__main__':
    from sklearn.datasets import fetch_openml

    boston = fetch_openml(name="house_prices", as_frame=True)
    X, y = boston.data, boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(len(X_train))
    print(len(X_test))
    print(len(y_train))
    print(len(y_test))
