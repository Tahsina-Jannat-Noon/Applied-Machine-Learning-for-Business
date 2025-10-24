# K-Nearest Neighbors (KNN) Classifier — Iris Dataset

## Project Overview
This project implements and explains the **K-Nearest Neighbors (KNN)** classification algorithm using the **Iris dataset**.  
It covers the full machine learning workflow — from **data preprocessing** and **feature scaling** to **model tuning**, **evaluation**, and **visualization** with the goal of understanding how KNN works in practice.



## Objective
To understand and demonstrate how the **KNN algorithm** classifies data points based on similarity and distance, using a real-world dataset.



## 📊 Dataset
**Dataset:** [Iris Dataset (scikit-learn)](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)  
- 150 samples across 3 flower species: *Setosa*, *Versicolor*, and *Virginica*  
- 4 features: Sepal Length, Sepal Width, Petal Length, and Petal Width  

The dataset is clean, balanced, and ideal for testing distance-based algorithms.



## Project Workflow
1. **Data Preprocessing**  
   - Split into Train (60%), Validation (20%), and Test (20%) sets.  
   - Added small Gaussian noise to improve generalization.  
   - Applied `StandardScaler` for feature standardization.  

2. **Model Training & Optimization**  
   - Used `GridSearchCV` to tune `n_neighbors`, `weights`, and `metric`.  
   - Combined Train and Validation sets for cross-validated optimization.  

3. **Evaluation**  
   - Achieved **93.3% test accuracy** and **96.7% ± 4.5% cross-validation accuracy**.  
   - Assessed performance with **confusion matrix**, **classification report**, and **ROC curves**.  
   - Compared against a **Dummy Classifier baseline (33.3%)**.  

4. **Model Saving**  
   - Saved the final model as `optimized_knn_iris.pkl` using **Joblib**.



## Results Summary

| Metric | Result |
|--------|---------|
| **Best Parameters** | `k=5`, `metric='euclidean'`, `weights='uniform'` |
| **Cross-Validation Accuracy** | 96.7% ± 4.5% |
| **Test Accuracy** | 93.3% |
| **Baseline Accuracy** | 33.3% |
| **AUC (All Classes)** | ~0.99 |

**Conclusion:**  
The optimized KNN classifier performed exceptionally well on the Iris dataset, showing strong generalization, high accuracy, and excellent class separability.



## Tools & Libraries
- Python  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- joblib  



