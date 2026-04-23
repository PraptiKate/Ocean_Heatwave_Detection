# 🌊 Ocean Heatwave Detection 

## 📌 Project Overview

This project implements an AI-based system to detect and predict marine heatwaves using oceanographic and environmental data. The model is built from scratch using the Delta Rule (Widrow-Hoff Learning Algorithm), a supervised learning approach that updates weights using gradient-based optimization.


---

## 🎯 Objective

* Detect marine heatwave conditions from ocean data
* Analyze environmental patterns affecting oceans
* Build machine learning models from scratch
* Compete using a real-time leaderboard

---

## 🏆 Live Leaderboard

🚀 Track model performance in real-time:

👉 **[🌐 View Leaderboard](https://praptikate.github.io/Ocean_Heatwave_Detection/)**

---

## 🚀 How to Participate

### 1️⃣ Fork the Repository

Click the **Fork** button (top right) to create your own copy.

---

### 2️⃣ Add Your Model

Add your model inside the `submissions/` folder:

```
submissions/your_name_model.py
```

---

### 3️⃣ Create a Pull Request (PR)

* Go to your forked repo
* Click **Compare & Pull Request**
* Submit your PR

---

### 4️⃣ Automatic Evaluation

Once your PR is created:

* Your model is evaluated automatically
* Accuracy & F1 Score are calculated
* Your name is added to the leaderboard

---

## 📂 Submission Format

Your submission must include this function:

```
def predict(X_train, y_train, X_test):
    return predictions
```

---

## 📜 Rules

* Do not modify `evaluate.py`
* Use the provided dataset only
* Follow the submission format strictly
* Submit only one model per PR

---

## ⚙️ Evaluation Process

Each Pull Request triggers an automated workflow:

1. Dataset is loaded
2. Your model is loaded
3. `predict()` function is executed
4. Performance metrics are calculated:

   * Accuracy
   * F1 Score
5. Leaderboard is updated automatically

---

## 🤝 Contributing

* Fork the repository
* Add your model
* Submit a Pull Request
* Check leaderboard for results

---

## 📊 Dataset

**File Used:** `realistic_ocean_climate_dataset.csv`

### Features:
- Location (encoded)
- Sea Surface Temperature (SST °C)
- pH Level
- Bleaching Severity
- Species Observed

### Dropped Features:
- Date  
- Latitude  
- Longitude  

### Target Variable:
- Marine Heatwave  
  - 1 → Heatwave present  
  - 0 → Normal condition  

---


### 6️⃣ Evaluation

Metrics used:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

Also includes sklearn classification_report for detailed analysis  

---

## 🧠 Model Insights
- Higher SST and lower pH are strong indicators of heatwaves  
- Bleaching severity has a direct relationship with heatwave occurrence  
- The model learns non-linear relationships using sigmoid activation  

---

## 📦 Libraries Used
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## 🔥 Key Highlights
- Machine learning model built from scratch (Delta Rule)  
- End-to-end pipeline: preprocessing → training → evaluation  
- Real-world application in climate and marine monitoring  
- Strong visualizations and interpretability  

---


## 🌍 Impact
This system can help in:
- Climate change monitoring  
- Marine ecosystem protection  
- Early detection of marine heatwaves  

---

## 👨‍💻 Author
Developed as a Data Science project focusing on AI applications in environmental systems.
