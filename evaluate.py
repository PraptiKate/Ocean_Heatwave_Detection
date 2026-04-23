import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Step 1: username
username = os.getenv("GITHUB_ACTOR", "unknown")

# Step 2: load dataset
df = pd.read_csv("Data/data.csv")

# Step 3: preprocessing
df = df.drop(columns=['Date', 'Latitude', 'Longitude'])

le = LabelEncoder()
df['Location'] = le.fit_transform(df['Location'])

bleach_map = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3}
df['Bleaching Severity'] = df['Bleaching Severity'].fillna('None')
df['Bleaching Severity'] = df['Bleaching Severity'].map(bleach_map)

df['Marine Heatwave'] = df['Marine Heatwave'].astype(int)

# Step 4: split
X = df.drop(columns=['Marine Heatwave']).values.astype(np.float32)
y = df['Marine Heatwave'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# add bias
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

# Step 6: Delta Rule Model
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

# Step 7: train model
model = DeltaRule(X_train.shape[1])
model.train(X_train, y_train)

# Step 8: evaluate
y_pred = model.predict(X_test)

accuracy = round(accuracy_score(y_test, y_pred), 3)
f1 = round(f1_score(y_test, y_pred), 3)

# Step 9: save result
result = {
    "name": username,
    "accuracy": accuracy,
    "f1": f1
}

with open("result.json", "w") as f:
    json.dump(result, f)
