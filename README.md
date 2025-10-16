# Rainfall Prediction Using Machine Learning

This project predicts rainfall occurrence using machine learning techniques. The goal is to analyze weather-related data, preprocess it, and train a model to classify whether it will rain based on input features.

* View the Project

## About

The project uses a dataset containing weather parameters such as temperature, humidity, wind speed, and rainfall measurements to predict rainfall. The workflow covers data cleaning, preprocessing, feature engineering, model training, and evaluation.

All analyses are performed in **Jupyter Notebook** for step-by-step execution.

## Tools & Technologies Used

* **Python (3.x)**
* **Pandas** – for data cleaning and manipulation
* **NumPy** – for numerical computations
* **Matplotlib / Seaborn** – for visualization
* **scikit-learn (sklearn)** – for preprocessing, modeling, and evaluation
* **SMOTE (from imbalanced-learn)** – to balance the dataset
* **Jupyter Notebook** – for interactive analysis

## Data Cleaning & Preprocessing

1. **Handle Missing Values and Duplicates**

   * Checked for null values using `isnull().sum()` and removed duplicates.

2. **Dataset Overview**

   * Explored `shape`, `dtypes`, `columns`, `nunique`, `info()`, and `describe()` for statistical understanding.

3. **Outlier Detection and Removal**

   * Identified outliers using **Z-score** and **box plots** and removed them to improve model accuracy.

4. **Scaling & Skewness Handling**

   * Separated numerical columns and examined skewness.
   * Applied **StandardScaler** to reduce bias in input variables.

5. **Feature Engineering**

   * Converted selected numerical columns into categorical columns if needed.
   * Performed **Label Encoding** on categorical columns using sklearn’s `LabelEncoder`.

6. **Balancing Dataset**

   * Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset and avoid bias towards majority class.

7. **Train-Test Split**

   * Split the dataset into training and testing sets using `train_test_split`.

## Modeling & Evaluation

1. **Model Used**

   * Logistic Regression from scikit-learn.

2. **Training the Model**

   * Fitted the Logistic Regression model on the training data.

3. **Evaluation Metrics**

   * Accuracy Score (`accuracy_score`)
   * Confusion Matrix (`confusion_matrix`)
   * Classification Report (`classification_report`)
   * ROC Curve (`roc_curve`)

4. **Cross-Validation**

   * Applied **K-Fold Cross Validation** to validate model generalization using `cross_val_score`.

## Key Insights

* Data cleaning, outlier removal, and scaling improved the model’s prediction performance.
* Balancing the dataset using SMOTE prevented bias in predictions.
* Logistic Regression effectively classified rainfall occurrence with reliable metrics.
* Label encoding of categorical variables ensured model compatibility and improved feature processing.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for rainfall prediction: from cleaning and preprocessing data, handling outliers, scaling, and balancing the dataset to training, evaluating, and validating a predictive model.



