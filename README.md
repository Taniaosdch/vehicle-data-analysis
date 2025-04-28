# 🚗 Predicting used car selling prices (Data Science Project)

This project builds a machine learning solution for predicting used car selling prices to support dynamic pricing strategies. 

The data was taken from: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho


## Problem Statement

Used car prices vary significantly based on features such as age, mileage, fuel type, transmission, and seller type.  
The goal is to predict the **selling price** based on car attributes to enable dynamic pricing.


## Project Structure

- **Data Analysis & Cleaning**  
  - Exploratory Data Analysis (EDA)
  - Handling categorical and numerical features
  - Feature engineering (e.g., creating `car_age` from `Year`)

- **Preprocessing Pipelines**  
  - Scaling numerical features
  - One-hot encoding categorical features
  - Using `ColumnTransformer` and `Pipeline` for clean workflows

- **Model Building**  
  - Baseline models: Linear Regression, Ridge, Lasso, Random Forest, XGBoost
  - Model evaluation using MAE, RMSE, and R²

- **Model Tuning**  
  - Hyperparameter optimization using RandomizedSearchCV
  - Selected best models based on RMSE

- **Final Model**  
  - Tuned **XGBoost Regressor** achieved the best performance (RMSE ≈ ±1230 units)
  - Model saved for reuse with `joblib`

- **Visualizations**  
  - Feature importance plot
  - Predicted vs Actual prices
  - Residual analysis


## 📈 Final Results

| Model           | RMSE   | R²    |
|:----------------|:------:|:-----:|
| Random Forest   | 2.01   | 0.96  |
| XGBoost (tuned) | 1.23   | 0.96  |

- Final selected model: **XGBoost Regressor**
- Model predicts selling prices within an average deviation of **±1230 units**.

---

## 📦 Requirements

- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib
  - seaborn
  - joblib

Install dependencies using:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
