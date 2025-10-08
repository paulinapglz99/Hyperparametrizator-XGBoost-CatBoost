# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from hyperopt import fmin, tpe, hp
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK
from google.colab import drive

"""# Get data"""

drive.mount('/content/drive')

!ls "/content/drive/My Drive/dreamAD"

#Read file
file_path = '/content/drive/My Drive/dreamAD/final_datasets/dataset_mtg.csv'
data = pd.read_csv(file_path)

data.head()

#Set targets
target_cols = ['Thal', 'Braak', 'CERAD', 'ADNC',
               'percent 6e10 positive area',
               'percent AT8 positive area',
               'percent NeuN positive area',
               'percent GFAP positive area',
               'percent aSyn positive area',
               'percent pTDP43 positive area']

#Check data to identify any inconsistency like NaNs
print("Information about the target columns:\n")
for col in target_cols:
    dtype = data[col].dtype
    non_null_count = data[col].count()
    print(f"Column: '{col}'")
    print(f"Data type: {dtype}")
    print(f"# of non-nulls: {non_null_count}\n")

#Exclude the target variables AND the “Donor ID” column from the feature table
columns_to_drop = target_cols + ['Donor ID']
X_features = data.drop(columns=columns_to_drop, errors='ignore') #The `errors=“ignore”` is useful if the column does not exist in the DataFrame, avoiding an error.

#Define the parameters for hyperparameterisation
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

#List for storing hyperparameter results
results = []

#Initialise the model and GridSearchCV
model = xgb.XGBRegressor(objective='reg:squarederror')
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           cv=7,
                           n_jobs=-1,
                           verbose=1)

#Loop for each target variable
for target_name in target_cols:
    print(f"--- Hyperparameterisation for the target variable: {target_name} ---")

    #Set target name
    y = data[target_name]

    valid_indices = y.dropna().index

    if len(valid_indices) < 2:
        print(f"There is insufficient data for {target_name}. Skipping this variable.")
        continue

    X_cleaned = X_features.loc[valid_indices]
    y_cleaned = y.loc[valid_indices]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)
    except ValueError as e:
        print(f"Error en train_test_split for {target_name}: {e}. Skipping.")
        continue

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    row = {'target': target_name, 'best_score': best_score}
    row.update(best_params)
    results.append(row)

    print(f"Best parameters for  {target_name}: {best_params}")
    print(f"Best cross-validation score for {target_name}: {best_score:.4f}\n")

hyper_df = pd.DataFrame(results)
print("Hyperparameterisation results saved in object 'hyper_df'")

#hyper_df

import os
os.path.join("/content/drive/My Drive/dreamAD", "hiperparametros_optimizados_mtg.csv")

results_df.to_csv("/content/drive/My Drive/dreamAD/hiperparametros_optimizados_mtg.csv", index=False)
