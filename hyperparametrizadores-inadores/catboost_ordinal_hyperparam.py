# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1dVYL2JYKa72Hfq-8zOv12AYhNzNr-7IU
"""

#pip install catboost

import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"

import numpy as np
np.int = int
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from catboost import CatBoostRegressor
#from google.colab import drive
from catboost import CatBoostRegressor

#drive.mount('/content/drive')

#Read file
file_path = '~/DreamAD/dataset_a9.csv'
data = pd.read_csv(file_path)

#QWK function
def qwk_metric(y_true, y_pred):
    y_pred_rounded = np.round(y_pred).astype(int)
    y_true_int = np.round(y_true).astype(int)
    return cohen_kappa_score(y_true_int, y_pred_rounded, weights='quadratic')

#Target columns
target_cols = [
    'Thal', 'Braak', 'CERAD', 'ADNC', 'LEWY', 'LATE'    #
]

drop_cols = ['percent 6e10 positive area',
             'percent AT8 positive area',
             'percent NeuN positive area',
             'percent GFAP positive area',
             'percent aSyn positive area',
             'percent pTDP43 positive area']

#Inspect data
print("Information about the target columns:\n")
for col in target_cols:
    dtype = data[col].dtype
    non_null_count = data[col].count()
    print(f"Column: '{col}'")
    print(f"Data type: {dtype}")
    print(f"# of non-nulls: {non_null_count}\n")

#Define features
columns_to_drop = target_cols + drop_cols + ['Donor ID']
X_features = data.drop(columns=columns_to_drop, errors='ignore')

#Define hyperparameter search space
space = {
    'iterations': hp.choice('iterations', [200, 400, 600]),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
    'depth': hp.choice('depth', [4, 6, 8]),
    'l2_leaf_reg': hp.choice('l2_leaf_reg', [1, 3, 5]),
    'subsample': hp.uniform('subsample', 0.7, 0.9),
    'rsm': hp.uniform('rsm', 0.7, 0.9)
}

#Loop through target variables
results = []

for target_name in target_cols:
    print(f"\n=== Optimizing for target: {target_name} ===")

    y = data[target_name]
    valid_indices = y.dropna().index

    if len(valid_indices) < 2:
        print(f"Insufficient data for {target_name}. Skipping.")
        continue

    X_cleaned = X_features.loc[valid_indices]
    y_cleaned = y.loc[valid_indices]

    X_train, X_test, y_train, y_test = train_test_split(
        X_cleaned, y_cleaned, test_size=0.2, random_state=42
    )

    def objective(params):
        params_fixed = {
            'iterations': int(params['iterations']),
            'depth': int(params['depth']),
            'l2_leaf_reg': int(params['l2_leaf_reg']),
            'learning_rate': float(params['learning_rate']),
            'subsample': float(params['subsample']),
            'rsm': float(params['rsm'])
        }

        model = CatBoostRegressor(
            loss_function='RMSE',
            random_seed=42,
            thread_count=6,
            verbose=False,
            **params_fixed
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        qwk = qwk_metric(y_test, y_pred)
        return {'loss': -qwk, 'status': STATUS_OK}

    #
    trials = Trials()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=25,
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    best_params = {
        'iterations': [200, 400, 600][best['iterations']],
        'depth': [4, 6, 8][best['depth']],
        'l2_leaf_reg': [1, 3, 5][best['l2_leaf_reg']],
        'learning_rate': best['learning_rate'],
        'subsample': best['subsample'],
        'rsm': best['rsm']
    }

    best_model = CatBoostRegressor(
        loss_function='RMSE',
        random_seed=42,
        thread_count=10,
        verbose=False,
        **best_params
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    best_qwk = qwk_metric(y_test, y_pred)

    results.append({
        'target': target_name,
        'best_qwk': best_qwk,
        **best_params
    })

    print(f"Best QWK for {target_name}: {best_qwk:.4f}")
    print(f"Best params: {best_params}")

hyper_df = pd.DataFrame(results)
print("Hyperparameterisation results saved in object 'hyper_df'")

#hyper_df

#Final save
hyper_df.to_csv("~/DreamAD/hyperparams_final/CatBoost-hiperparametros_ordinales_optimizados_a9.csv", index=False)

print("\nHyperparameter optimization completed and results saved.")
