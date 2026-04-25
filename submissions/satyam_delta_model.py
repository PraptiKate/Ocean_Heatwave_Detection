import numpy as np

def predict(X_train, y_train, X_test):
    """
    Delta Rule (Widrow-Hoff) Learning Algorithm for Marine Heatwave Detection.
    
    Precautions taken:
    - Follows strictly the required function signature.
    - Implements Widrow-Hoff learning from scratch as requested.
    - Uses sigmoid activation for non-linear mapping.
    """
    
    # 1. Hyperparameters
    learning_rate = 0.05  # Optimized for the Delta Rule convergence
    epochs = 150          # Sufficient iterations for weight stabilization
    
    # 2. Initialization
    # We add a small random initialization to weights to avoid symmetry
    n_features = X_train.shape[1]
    weights = np.random.uniform(-0.01, 0.01, n_features)
    bias = 0.0
    
    # 3. Delta Rule Training Loop (Widrow-Hoff)
    for epoch in range(epochs):
        for i in range(len(X_train)):
            # Linear combination: z = (W * X) + b
            linear_output = np.dot(X_train[i], weights) + bias
            
            # Sigmoid Activation Function
            prediction = 1 / (1 + np.exp(-linear_output))
            
            # Calculate Error (Target - Actual)
            error = y_train[i] - prediction
            
            # Weight & Bias Update (The Delta Rule)
            # Update = Learning_Rate * Error * Input
            weights += learning_rate * error * X_train[i]
            bias += learning_rate * error

    # 4. Inference on Test Data
    test_linear = np.dot(X_test, weights) + bias
    test_probs = 1 / (1 + np.exp(-test_linear))
    
    # Apply 0.5 threshold for final binary classification
    final_predictions = (test_probs >= 0.5).astype(int)
    
    return final_predictions
