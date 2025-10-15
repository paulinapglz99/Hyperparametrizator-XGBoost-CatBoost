# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1PMoojs7sSNUrz51nMM5_6WwBP4VOklid
"""
#Import packages
import os
# Limitar número de hilos en todo el entorno
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"

import numpy as np
np.int = int
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score #QWK
from hyperopt import fmin, tpe, hp, STATUS_OK
#from sklearn.metrics import accuracy_score
#from google.colab import drive
#from catboost import CatBoostRegressor
#np.int = int

import xgboost as xgb
xgb.set_config(nthread=10)
print(xgb.get_config())


"""# Set functions"""

#Concordance Correlation Coefficient (CCC)
def ccc_metric(y_true, y_pred):
    """Concordance Correlation Coefficient"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc

"""# Get data"""

#drive.mount('/content/drive')

#!ls "/content/drive/My Drive/dreamAD/hiperaparametros-final/"

#Read file
file_path = '~/DreamAD/dataset_a9.csv'
data = pd.read_csv(file_path)

data.head()

#Set targets
target_cols = [#'Thal', 'Braak', 'CERAD', 'ADNC', 'LEWY', 'LATE'#,
               'percent 6e10 positive area',
               'percent AT8 positive area',
               'percent NeuN positive area',
               'percent GFAP positive area',
               'percent aSyn positive area',
               'percent pTDP43 positive area'
               ]

drop_cols = ['Thal', 'Braak', 'CERAD', 'ADNC', 'LEWY', 'LATE']

#Check data to identify any inconsistency like NaNs
print("Information about the target columns:\n")
for col in target_cols:
    dtype = data[col].dtype
    non_null_count = data[col].count()
    print(f"Column: '{col}'")
    print(f"Data type: {dtype}")
    print(f"# of non-nulls: {non_null_count}\n")

#Exclude the target variables AND the “Donor ID” column from the feature table
columns_to_drop = target_cols + drop_cols + ['Donor ID']
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
model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=10)
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           cv=7,
                           n_jobs=1,
                           verbose=1,
                           scoring='neg_mean_squared_error')  #We use MSE as a proxy, then evaluate QWK manually.

#Loop for each target variable
for target_name in target_cols:
    print(f"--- Hyperparameterisation for the target variable: {target_name} ---")

    #Set target
    y = data[target_name]

    valid_indices = y.dropna().index

    if len(valid_indices) < 2:
        print(f"There is insufficient data for {target_name}. Skipping this variable.")
        continue

    X_cleaned = X_features.loc[valid_indices]
    y_cleaned = y.loc[valid_indices]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_cleaned, y_cleaned, test_size=0.2, random_state=42
        )
    except ValueError as e:
        print(f"Error en train_test_split for {target_name}: {e}. Skipping.")
        continue

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Evaluación final con CCC
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    ccc_score = ccc_metric(y_test, y_pred)

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # Guardar resultados incluyendo CCC
    row = {
        'target': target_name,
        'best_cv_score_MSE': best_cv_score,
        'best_CCC': ccc_score
    }
    row.update(best_params)
    results.append(row)

    print(f"Best parameters for {target_name}: {best_params}")
    print(f"Best CCC for {target_name}: {ccc_score:.4f}\n")

# Convertir resultados a DataFrame
hyper_df = pd.DataFrame(results)
print("Hyperparameterisation results saved in object 'hyper_df'")

#hyper_df

#Save results
hyper_df.to_csv("~/DreamAD/hyperparams_final/XGBoost-hiperparametros_continuos_optimizados_a9.csv", index=False)
