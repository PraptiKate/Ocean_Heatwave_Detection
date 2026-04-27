import numpy as np

def predict(X_train, y_train, X_test):
    """
    Improved Delta Rule Model (Better Accuracy Version)
    """

    # 1. Hyperparameters
    learning_rate = 0.1
    epochs = 300

    # 2. Feature Scaling (VERY IMPORTANT)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test  = (X_test - mean) / std

    # 3. Add Bias Column
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test  = np.hstack([X_test,  np.ones((X_test.shape[0], 1))])

    # 4. Initialization
    np.random.seed(42)
    weights = np.random.randn(X_train.shape[1]) * 0.01

    # 5. Sigmoid Function
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # 6. Training (Vectorized Delta Rule)
    for epoch in range(epochs):

        # Forward pass
        z = X_train @ weights
        y_pred = sigmoid(z)

        # Error
        error = y_train - y_pred

        # Sigmoid derivative
        grad = error * y_pred * (1 - y_pred)

        # Weight update (batch gradient)
        weights += learning_rate * (X_train.T @ grad) / len(X_train)

    # 7. Prediction
    z_test = X_test @ weights
    probs = sigmoid(z_test)

    return (probs >= 0.5).astype(int)
