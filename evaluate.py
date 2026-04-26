import glob
import importlib.util
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("Data/data.csv")

# Preprocessing
df = df.drop(columns=['Date', 'Latitude', 'Longitude'])

le = LabelEncoder()
df['Location'] = le.fit_transform(df['Location'])

bleach_map = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3}
df['Bleaching Severity'] = df['Bleaching Severity'].fillna('None')
df['Bleaching Severity'] = df['Bleaching Severity'].map(bleach_map)

df['Marine Heatwave'] = df['Marine Heatwave'].astype(int)

# Split
X = df.drop(columns=['Marine Heatwave']).values.astype(np.float32)
y = df['Marine Heatwave'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add bias
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

# Load all submissions
files = glob.glob("submissions/*.py")

if len(files) == 0:
    raise Exception("No submission file found")

results = []

for submission_file in files:
    try:
        spec = importlib.util.spec_from_file_location("model", submission_file)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        if not hasattr(model_module, "predict"):
            continue

        y_pred = model_module.predict(X_train, y_train, X_test)
        y_pred = np.array(y_pred)

        if len(y_pred) != len(y_test):
            continue

        accuracy = round(accuracy_score(y_test, y_pred), 3)
        f1 = round(f1_score(y_test, y_pred), 3)

        name = os.path.basename(submission_file).replace(".py", "")

        results.append({
            "name": name,
            "accuracy": accuracy,
            "f1": f1
        })

        print(f"{name}: Acc={accuracy}, F1={f1}")

    except Exception as e:
        print(f"Error in {submission_file}: {e}")

# Save result.json (ONLY this, no CSV here)
with open("result.json", "w") as f:
    json.dump(results, f)
