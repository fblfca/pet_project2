import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
from time import time

from sklearn.pipeline import (
    Pipeline, make_pipeline
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, train_test_split,
    KFold, StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import optuna
import joblib

SEED = 67
WORK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = WORK_DIR / 'data' / 'interim'

def data_load(path):
    '''Функция предназначена для загрузки предобработанных данных из скрипта validation.py'''
    df = pd.read_csv(path / 'interim_hotel_bookings.csv', encoding='utf-8').drop_duplicates().reset_index(drop=True)
    return df

def data_split(df):
    '''Разделение данных на train, validation и test выборок'''
    y = df['is_canceled']
    X = df.drop(columns=['is_canceled', 'reservation_status_date', 'booking_changes', 'days_in_waiting_list', 'assigned_room_type'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )

    train_size = X_train.shape[0] // 6

    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train,
        test_size=0.2,
        train_size=train_size,
        stratify=y_train,
        random_state=SEED,
    )

    return (
        X,
        y,
        X_sample,
        y_sample,
        X_train,
        y_train,
        X_test,
        y_test
    )

X, y, X_sample, y_sample, X_train, y_train, X_test, y_test = (
        data_split(data_load(DATA_DIR))
)
numeric = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
    'previous_cancellations', 'previous_bookings_not_canceled', 'adr', 'required_car_parking_spaces',
    'total_of_special_requests'
]
categorical = X.columns.difference(numeric).tolist()

def cust_pipeline(model) -> Pipeline:
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encoding', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=50, sparse_output=True))
    ])

    preprocessing = ColumnTransformer([
        ('numeric', num_pipe, numeric),
        ('categorical', cat_pipe, categorical)
    ])

    return Pipeline([('preprocessing', preprocessing), ('model', model)])

def model_objective(trial):
    model = trial.suggest_categorical('model', [
        'CatBoostClassifier', 'RandomForestClassifier'
    ])

    if model == 'CatBoostClassifier':
        return cust_pipeline(CatBoostClassifier(
            iterations=trial.suggest_int('iterations', 200, 700),
            depth=trial.suggest_int('depth', 4, 20),
            learning_rate=trial.suggest_float(
                'learning_rate', 0.01, 0.2, log=True
            ),
            verbose=False,
            random_seed=SEED,
            thread_count=1,
        ))

    return cust_pipeline(RandomForestClassifier(
                criterion=trial.suggest_categorical('criterion', [
                    'gini', 'log_loss', 'entropy'
                ]),
                max_depth=trial.suggest_int('max_depth', 4, 20),
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 2, 50),
                random_state=SEED,
                n_jobs=-1
            ))

def objective(trial):
    score = cross_val_score(
        model_objective(trial),
        X_sample,
        y_sample,
        scoring='average_precision',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        verbose=-1
    )

def get_best_params():
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(),
        direction='maximize'
    )
    study.optimize(
        objective,
        n_trials=50,
        n_jobs=1
    )

    return study.best_trial.params

def final_model(params: dict):
    name = params['model']
    
    if name == 'RandomForestClassifier':
        return cust_pipeline(RandomForestClassifier(
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=SEED,
            n_jobs=-1
        ))
    
    return cust_pipeline(CatBoostClassifier(
        iterations=params['iterations'],
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        verbose=False,
        random_seed=SEED,
        thread_count=1,
    ))

def main():
    start_time = time()

    best_params = get_best_params()
    model = final_model(best_params)

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1': f1_score(y_test, predictions, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probabilities),
        'pr_auc': average_precision_score(y_test, probabilities),
    }

    artifacts_dir = WORK_DIR / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    artifacts_model = artifacts_dir / 'best_pipeline.joblib'
    joblib.dump(model, artifacts_model)

    print('Лучшие параметры:')
    print(best_params)

    print('\nМетрики на тестовой выборке:')
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')

    print(f'\nВремя выполнения: {time() - start_time:.1f} сек.')

    return {
        'model': model,
        'best_params': best_params,
        'metrics': metrics,
        'model_path': artifacts_model
    }


if __name__ == '__main__':
    results = main()
