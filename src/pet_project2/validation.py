import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


WORK_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = WORK_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

RAW_FILE = RAW_DATA_DIR / 'hotel_bookings.csv'

MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Исходный файл не найден: {path}')

    df = pd.read_csv(path, encoding='utf-8')

    if df.empty:
        raise ValueError(f'В файле нет данных: {path}')

    required_columns = {
        'children',
        'agent',
        'company',
        'reservation_status_date',
        'arrival_date_month',
        'country',
        'market_segment',
        'distribution_channel',
        'reservation_status',
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f'В исходном файле отсутствуют столбцы: {sorted(missing_columns)}'
        )

    return df


def nan_formatting(df: pd.DataFrame) -> pd.DataFrame:
    '''Привести текстовые маркеры пропусков к NaN.'''
    return df.replace('Undefined', np.nan)


def datatype_formatting(df: pd.DataFrame) -> pd.DataFrame:
    '''Преобразовать отдельные столбцы к нужным типам.'''
    df = df.copy()

    numeric_columns = ['children', 'agent', 'company']

    df[numeric_columns] = (
        df[numeric_columns]
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
        .astype('int64')
    )

    df['reservation_status_date'] = pd.to_datetime(
        df['reservation_status_date'],
        errors='coerce',
        format='%Y-%m-%d',
    )

    return df


def nan_imputation(df: pd.DataFrame) -> pd.DataFrame:
    '''Заполнить пропуски в выбранных категориальных столбцах.'''
    df = df.copy()

    columns = ['country', 'market_segment', 'distribution_channel']
    imputer = SimpleImputer(strategy='most_frequent')
    df[columns] = imputer.fit_transform(df[columns])

    return df


def label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    '''Преобразовать строковые категориальные признаки в целые числа.'''
    df = df.copy()

    label_columns = (
        df.select_dtypes(include=['object', 'string'])
        .columns
        .drop('arrival_date_month', errors='ignore')
    )

    month_encoder = OrdinalEncoder(
        categories=[MONTHS],
        handle_unknown='use_encoded_value',
        encoded_missing_value=-1,
        unknown_value=-1,
    )

    category_encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        encoded_missing_value=-1,
        unknown_value=-1,
    )

    df['arrival_date_month'] = (
        month_encoder
        .fit_transform(df[['arrival_date_month']])
        .ravel()
        .astype('int64')
        + 1
    )

    if len(label_columns) > 0:
        df[label_columns] = (
            category_encoder
            .fit_transform(df[label_columns])
            .astype('int64')
        )

    return df


def main() -> None:
    df = load_data(RAW_FILE)

    # reservation_status раскрывает итог бронирования, поэтому исключаем его.
    df = df.drop(columns=['reservation_status'])

    df_int = (
        df.copy()
        .pipe(nan_formatting)
        .pipe(datatype_formatting)
    )

    df_prep = (
        df_int.copy()
        .pipe(nan_imputation)
        .pipe(label_encoding)
    )

    if df_prep.isna().any().any():
        null_counts = df_prep.isna().sum()
        null_counts = null_counts[null_counts > 0]
        raise ValueError(
            f'После подготовки остались пропуски:\n{null_counts}'
        )

    interim_path = INTERIM_DATA_DIR / 'interim_hotel_bookings.csv'
    processed_path = PROCESSED_DATA_DIR / 'processed_hotel_bookings.csv'

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_int.to_csv(interim_path, index=False, encoding='utf-8')
    df_prep.to_csv(processed_path, index=False, encoding='utf-8')

    print(f'Промежуточный файл сохранён: {interim_path}')
    print(f'Обработанный файл сохранён: {processed_path}')


if __name__ == '__main__':
    main()