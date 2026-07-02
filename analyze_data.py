from pathlib import Path
import pandas as pd

DATA_DIR = Path('data')
DATA_FILE = 'inventory_demand_forecasting_dataset.csv'  # Update this file name to your dataset file


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    data_path = DATA_DIR / DATA_FILE
    print(f'Loading data from: {data_path}')

    if not data_path.exists():
        raise FileNotFoundError(
            f'Data file not found. Please place your dataset in {DATA_DIR} and update DATA_FILE in analyze_data.py.'
        )

    df = load_data(data_path)
    print('Shape:', df.shape)
    print('\nColumns:')
    print(df.columns.tolist())
    print('\nInfo:')
    print(df.info())
    print('\nDescribe:')
    print(df.describe(include='all'))


if __name__ == '__main__':
    main()
