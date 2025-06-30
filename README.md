# cmapss-predictive-maintenance-ml
# Predictive Maintenance of Aircraft Engines using Machine Learning

This project implements a complete machine learning pipeline for **predictive maintenance** using the **NASA CMAPSS dataset**. The objective is to classify aircraft engine health into three categories: **Healthy**, **Warning**, and **Imminent Failure**, based on sensor data and time-series feature engineering.

## 📌 Project Overview

Industrial machinery failures can cause significant downtime and cost. This research builds a predictive model that anticipates failures using advanced ML techniques, addresses class imbalance, and adds explainability using SHAP and LIME.

### 🛠 Key Features

- ✅ RUL (Remaining Useful Life) estimation converted to multi-class classification
- ✅ Class balancing using a **hypergraph-based sampling algorithm**
- ✅ Feature selection using **Random Forest Gini importance** and **Permutation Importance**
- ✅ ML Models: **AdaBoost**, **Random Forest**, and **XGBoost**
- ✅ Explainable AI with **SHAP** (global) and **LIME** (local)
- ✅ Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

## 📂 Dataset

**NASA CMAPSS (FD001 – FD004)**  
- Simulates degradation patterns of turbofan engines  
- Includes 21 sensor readings per time cycle per engine  
- Available at: [NASA CMAPSS Data Repository](https://www.nasa.gov)

## 🧠 Models Used

| Model        | Strengths |
|--------------|-----------|
| AdaBoost     | Handles imbalanced data, focuses on hard samples |
| Random Forest| Robust baseline, Gini-based feature importance |
| XGBoost      | Best accuracy with regularization and fast convergence |

## 📈 Results

The models showed promising performance in predicting the health state of aircraft engines. XGBoost consistently delivered the best overall accuracy across all four sub-datasets (FD001–FD004), particularly excelling in distinguishing between the warning and imminent failure stages.

AdaBoost and Random Forest also performed competitively, especially in identifying healthy engine conditions. All models benefited significantly from class balancing and feature selection, improving their precision and recall. SHAP and LIME analyses provided interpretability, making the predictions transparent and justifiable in real-world maintenance settings.

## 🔍 Explainability

- **SHAP** explains feature contributions across the entire dataset.
- **LIME** gives interpretable visualizations for individual predictions — critical in aviation safety.

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Harrish7619/cmapss-predictive-maintenance-ml.git
   cd cmapss-predictive-maintenance-ml
