# 🧠 Decision Trees & Random Forests — Classification Project

## 📌 Overview  
The objective is to implement **Decision Tree** and **Random Forest** models for a classification problem (Heart Disease dataset), analyze their performance, tune hyperparameters, and interpret results.  

The notebook includes **EDA, preprocessing, model training, tuning, evaluation, and visualizations** to make the analysis comprehensive and professional.

---

## 📊 Dataset
- **Source:** [Heart Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Target:** `target` — indicates presence (1) or absence (0) of heart disease.
- **Features:** Patient attributes like age, sex, blood pressure, cholesterol, max heart rate, etc.
- **Size:** ~300 rows, 14 columns.

---

## 🛠 Tools & Libraries
- Python 3.x
- Pandas, NumPy (data handling)
- Matplotlib, Seaborn (visualization)
- scikit-learn (ML models & evaluation)
- Graphviz / plot_tree (Decision Tree visualization)

---

## 🚀 Features Implemented
1. **Exploratory Data Analysis (EDA)**
   - Missing value detection & handling
   - Summary statistics
   - Correlation heatmap
   - Class distribution

2. **Data Preprocessing**
   - One-hot encoding for categorical features
   - Train-test split with stratification

3. **Model Training**
   - Baseline **Decision Tree Classifier**
   - Baseline **Random Forest Classifier**

4. **Overfitting Analysis**
   - Effect of `max_depth` on train/test accuracy

5. **Hyperparameter Tuning (GridSearchCV)**
   - Decision Tree: `max_depth`, `min_samples_split`, `min_samples_leaf`
   - Random Forest: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`

6. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-score
   - ROC AUC (binary classification)
   - Confusion Matrix heatmaps

7. **Model Interpretation**
   - Feature importance comparison for both models

8. **Cross-Validation**
   - k-fold CV accuracy scores

---

## 📈 Results Summary
| Model | Accuracy | Precision | Recall | F1-score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Decision Tree (tuned) | ~0.84 | ~0.83 | ~0.85 | ~0.84 | ~0.86 |
| Random Forest (tuned) | ~0.88 | ~0.88 | ~0.88 | ~0.88 | ~0.91 |

> **Observation:** Random Forest outperforms a single Decision Tree due to ensemble averaging, reducing variance and improving generalization.

---

## 📷 Visual Outputs
- Correlation heatmap
- Decision Tree (first 3 levels)
- Overfitting curve for Decision Tree
- Confusion matrices for both models
- ROC curves comparing AUC
- Feature importance bar plots

---

## 📝 Conclusion
- Random Forest consistently performed better than a single Decision Tree.
- Hyperparameter tuning and cross-validation significantly improved performance.
- Feature importance analysis helps in understanding model decisions.
- This approach can be generalized to other classification datasets.

---
