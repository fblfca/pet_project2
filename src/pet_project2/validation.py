import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path

RAW_DATA_DIR = Path('./data/raw')
INTERIM_DATA_DIR = Path('./data/interim')
PROCESSED_DATA_DIR = Path('./data/processed')

df = pd.read_csv(RAW_DATA_DIR / 'hotel_bookings.csv')

def nan_formatting(df: pd.DataFrame) -> pd.DataFrame:
    '''Функция, предназначенная для приведения всех пропусков к единому виду.
    Сейчас в датасете есть как и NaN-значения, так и "Undefined". '''

    df[df.values == 'Undefined'] = np.nan

    return df


def datatype_formatting(df: pd.DataFrame) -> pd.DataFrame:
    '''Функция, предназначенная для приведения колонок к корректному формату данных.
    Конкретнее: некоторые данные, имеющие целочисленный или дробный характер имеют
    строковый формат'''

    str_to_int: list[str] = ['children', 'agent', 'company']
    data_cols: str = 'reservation_status_date'

    df[str_to_int] = (df[str_to_int]
                      .apply(pd.to_numeric, errors='coerce')
                      .fillna(0).astype('int64'))

    df[data_cols] = pd.to_datetime(
        df[data_cols],
        errors='coerce',
        format='%Y-%m-%d'
    )

    return df

def nan_imputation(df: pd.DataFrame) -> pd.DataFrame:
    '''Функция, предназначенная для заполнения пропусков. Выбрана стратегия
    моды, ибо пропусков мало, на распределение данных это практически не повлияет.
    Статистической значимости в выборе способа заполнения как таковой нет.'''

    nan_to_moda = ['country', 'market_segment', 'distribution_channel']
    imputer = SimpleImputer(strategy='most_frequent')

    df[nan_to_moda] = imputer.fit_transform(df[nan_to_moda])

    return df

def label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    '''Функция, предназначенная для преобразования категориальных данных
    строкового типа в маркеры целочисленного формата. Это позволит не отбрасывать
    эти признаки при обучении ml-моделей.'''

    months = [
        'January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December'
    ]
    label_cols = (df.select_dtypes(include=['str', 'object'])
                  .columns
                  .drop('arrival_date_month', errors='ignore'))

    encoder1 = OrdinalEncoder(
        categories=[months],
        handle_unknown='use_encoded_value',
        encoded_missing_value=-1,
        unknown_value=-1
    )
    encoder2 = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        encoded_missing_value=-1,
        unknown_value=-1
    )

    df['arrival_date_month'] = (
        encoder1.fit_transform(df[['arrival_date_month']])
        .ravel()
        .astype('int64') + 1
    )

    df[label_cols] = encoder2.fit_transform(df[label_cols]).astype('int64')

    return df

df = df.drop(columns=['reservation_status'])

df_int = (
    df.copy()
    .pipe(nan_formatting)
    .pipe(datatype_formatting)
    .pipe(nan_imputation)
)

df_prep = (
    df_int.copy()
    .pipe(label_encoding)
)

int_output_path = INTERIM_DATA_DIR / 'interim_hotel_bookings.csv'
prep_output_path = PROCESSED_DATA_DIR / 'processed_hotel_bookings.csv'

INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

if int_output_path.exists():
    print(f'Файл уже существует: {int_output_path}')
else:
    df_int.to_csv(int_output_path, index=False, encoding='utf-8')

if prep_output_path.exists():
    print(f'Файл уже существует: {prep_output_path}')
else:
    df_prep.to_csv(prep_output_path, index=False, encoding='utf-8')