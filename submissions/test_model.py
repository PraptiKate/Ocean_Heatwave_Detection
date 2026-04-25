def predict(X_train, y_train, X_test):
    import numpy as np

    class DeltaRule:
        def __init__(self, n_inputs, lr=0.1):
            self.W = np.random.randn(n_inputs) * 0.01
            self.lr = lr

        def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))

        def predict_proba(self, X):
            return self.sigmoid(X @ self.W)

        def predict(self, X):
            return (self.predict_proba(X) >= 0.5).astype(int)

        def train(self, X, y, epochs=200):
            for _ in range(epochs):
                yhat = self.predict_proba(X)
                error = y - yhat
                grad = X.T @ (error * yhat * (1 - yhat))
                self.W += self.lr * grad / len(X)

    model = DeltaRule(X_train.shape[1])
    model.train(X_train, y_train)

    predictions = model.predict(X_test)
    return predictions
