import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Get username
username = os.getenv("GITHUB_ACTOR", "unknown")

# Step 2: Load dummy dataset (we improve later)
# For now using random data (structure ready)
y_true = np.random.randint(0, 2, 100)
y_pred = np.random.randint(0, 2, 100)

# Step 3: Calculate metrics
accuracy = round(accuracy_score(y_true, y_pred), 3)
f1 = round(f1_score(y_true, y_pred), 3)

# Step 4: Save result
result = {
    "name": username,
    "accuracy": accuracy,
    "f1": f1
}

with open("result.json", "w") as f:
    json.dump(result, f)
