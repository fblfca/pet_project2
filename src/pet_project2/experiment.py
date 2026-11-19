"""Финальное обучение LightGBM после сравнения моделей в ноутбуке 02."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / 'data' / 'raw' / 'hotel_bookings.csv'
REPORT_DIR = PROJECT_DIR / 'reports'
ARTIFACT_DIR = PROJECT_DIR / 'artifacts'
SEED = 42

# Параметры выбраны в ноутбуке 02 по StratifiedKFold.
BEST_PARAMS = {'num_leaves': 15, 'learning_rate': 0.05, 'n_estimators': 180}
LEAKAGE_COLS = [
    'reservation_status', 'reservation_status_date', 'assigned_room_type',
    'booking_changes', 'days_in_waiting_list'
]


def load_data():
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=LEAKAGE_COLS).replace('Undefined', np.nan)
    df['children'] = pd.to_numeric(df['children'], errors='coerce')
    for col in ['agent', 'company']:
        df[col] = df[col].astype('string').fillna('missing')
    y = df.pop('is_canceled').astype(int)
    return df, y


def make_pipeline(X):
    categorical = X.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric = X.columns.difference(categorical).tolist()
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=50,
                                  sparse_output=True))
    ])
    preprocessing = ColumnTransformer([
        ('numeric', numeric_pipe, numeric),
        ('categorical', categorical_pipe, categorical)
    ])
    model = LGBMClassifier(**BEST_PARAMS, random_state=SEED, n_jobs=4,
                           verbosity=-1)
    return Pipeline([('preprocessing', preprocessing), ('model', model)])


def get_metrics(y_true, probability, threshold):
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
    return {
        'accuracy': accuracy_score(y_true, predicted),
        'precision': precision_score(y_true, predicted),
        'recall': recall_score(y_true, predicted),
        'f1': f1_score(y_true, predicted),
        'roc_auc': roc_auc_score(y_true, probability),
        'pr_auc': average_precision_score(y_true, probability),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'assumed_loss': int(fn * 200 + fp * 20)
    }


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Порог подбираем на validation; test не участвует в этом выборе.
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )
    validation_model = make_pipeline(X_fit)
    validation_model.fit(X_fit, y_fit)
    valid_p = validation_model.predict_proba(X_valid)[:, 1]

    rows = []
    for threshold in np.arange(0.1, 0.91, 0.05):
        values = get_metrics(y_valid, valid_p, threshold)
        rows.append({'threshold': round(float(threshold), 2),
                     'fn': values['fn'], 'fp': values['fp'],
                     'assumed_loss': values['assumed_loss'],
                     'f1': values['f1']})
    costs = pd.DataFrame(rows)
    costs.to_csv(REPORT_DIR / 'threshold_costs.csv', index=False)
    threshold = float(costs.loc[costs['assumed_loss'].idxmin(), 'threshold'])
    f1_threshold = float(costs.loc[costs['f1'].idxmax(), 'threshold'])

    best_pipeline = make_pipeline(X_train)
    best_pipeline.fit(X_train, y_train)
    test_p = best_pipeline.predict_proba(X_test)[:, 1]
    metrics_default = get_metrics(y_test, test_p, 0.5)
    metrics_selected = get_metrics(y_test, test_p, threshold)
    metrics_f1 = get_metrics(y_test, test_p, f1_threshold)
    summary = {
        'best_model': 'LightGBM',
        'best_params': BEST_PARAMS,
        'training_rows_final': len(X_train),
        'test_rows': len(X_test),
        'selected_threshold': threshold,
        'f1_threshold': f1_threshold,
        'assumed_cost_fn': 200,
        'assumed_cost_fp': 20,
        'final_metrics': metrics_default,
        'final_metrics_selected_threshold': metrics_selected,
        'final_metrics_f1_threshold': metrics_f1
    }
    (REPORT_DIR / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    pd.DataFrame(confusion_matrix(y_test, test_p >= threshold),
                 index=['actual_0', 'actual_1'],
                 columns=['predicted_0', 'predicted_1']).to_csv(
                     REPORT_DIR / 'final_confusion_matrix.csv')
    joblib.dump(best_pipeline, ARTIFACT_DIR / 'best_model.joblib')
    print('Модель:', summary['best_model'])
    print('Порог:', threshold)
    print('Метрики при 0.5:', metrics_default)
    print('Метрики при выбранном пороге:', metrics_selected)
    print('Метрики при F1-пороге:', metrics_f1)


if __name__ == '__main__':
    main()
