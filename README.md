# Forest-cover-type-project
This project made by using machine learning models , and mainly the performed classification are gradient Boosting , and Random Forest  and KNN.
# 🌲 Forest Cover Type Prediction

## 📌 Project Overview

Forest Cover Type Prediction is a Machine Learning project that predicts the type of forest cover based on different geographical and terrain features.

The project uses a **Random Forest Classifier** to classify the forest into one of seven cover types.

A Streamlit frontend is also created where users can enter terrain details and get the predicted forest cover type.

---

## 🎯 Objectives

- Predict forest cover type using Machine Learning.
- Analyze different terrain and geographical features.
- Train a Random Forest classification model.
- Create a simple and interactive Streamlit frontend.
- Display the predicted forest cover type and prediction confidence.

---

## 🧠 Machine Learning Model

The project uses:

- **Random Forest Classifier**
- Number of trees: **100**
- Train-Test Split: **80% Training / 20% Testing**
- Random State: **42**
- Stratified splitting is used to maintain class distribution.

---

## 📊 Dataset

The project uses the **Forest Cover Type dataset**.

Important features include:

- Elevation
- Aspect
- Slope
- Horizontal Distance to Hydrology
- Vertical Distance to Hydrology
- Horizontal Distance to Roadways
- Horizontal Distance to Fire Points
- Other terrain and geographical features

### Feature Engineering

Two additional features are created:

1. **Distance To Hydrology**

```text
√(Horizontal Distance² + Vertical Distance²)
