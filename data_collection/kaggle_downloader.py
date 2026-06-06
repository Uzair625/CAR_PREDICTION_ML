"""
Kaggle dataset downloader.
Downloads 6 global used-car datasets covering USA, UK, Europe, India, and global markets.

Setup (one-time):
  1. Create a free Kaggle account at kaggle.com
  2. Go to Account -> API -> Create New Token  -> downloads kaggle.json
  3. Place kaggle.json at: C:/Users/<your-name>/.kaggle/kaggle.json  (Windows)
                        or: ~/.kaggle/kaggle.json                    (Linux/Mac)
  4. pip install kaggle

Then run:  python -m data_collection.kaggle_downloader
"""
import os
import zipfile
import shutil
import pandas as pd

RAW_DIR = os.path.join('data', 'raw')

# Dataset slug -> (filename_inside_zip, country_tag, currency, mileage_unit)
DATASETS = {
    # USA — 420K Craigslist listings (cars & trucks)
    'austinreese/craigslist-carstrucks-data': (
        'vehicles.csv', 'USA', 'USD', 'miles'
    ),
    # UK — 100K listings across 8 brands (BMW, Audi, Ford, Toyota, ...)
    'adityadesai13/used-car-dataset-100000-datapoints': (
        None, 'UK', 'GBP', 'miles'        # multiple CSVs merged below
    ),
    # India — CarDekho (extended version with 8K+ records)
    'nehalbirla/vehicle-dataset-from-cardekho': (
        'car data.csv', 'India', 'INR', 'km'
    ),
    # Global — 38K records from Russian/Eastern European catalogue
    'lepchenkov/usedcarscatalog': (
        'cars.csv', 'Global', 'USD', 'km'
    ),
    # USA — Vehicle sales from CarGurus (550K rows)
    'syedanwarafridi/vehicle-sales-data': (
        'car_prices.csv', 'USA', 'USD', 'miles'
    ),
    # India — Cars4U dataset (~7K records, richer features)
    'avikasliwal/used-cars-price-prediction': (
        'used_cars_data.csv', 'India', 'INR', 'km'
    ),
}

EXCHANGE_TO_USD = {
    'USD': 1.00, 'GBP': 1.27, 'EUR': 1.08,
    'INR': 0.012, 'AED': 0.272,
}

MILES_TO_KM = 1.60934


def _download(slug: str) -> str:
    """Download and unzip a Kaggle dataset. Returns the local folder path."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Kaggle package not installed. Run: pip install kaggle\n"
            "Then place your kaggle.json API key in ~/.kaggle/"
        )

    out_dir = os.path.join(RAW_DIR, slug.replace('/', '_'))
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Downloading {slug} ...")
    os.system(f'kaggle datasets download -d {slug} -p "{out_dir}" --unzip -q')
    return out_dir


def _normalize(df: pd.DataFrame, country: str, currency: str, mileage_unit: str) -> pd.DataFrame:
    """Map any dataset's columns to the standard schema."""
    col = {c.lower().strip(): c for c in df.columns}

    def get(candidates):
        for c in candidates:
            if c in col:
                return df[col[c]]
        return pd.Series([None] * len(df))

    name         = get(['name', 'model', 'car_name', 'title', 'vehicle'])
    company      = get(['company', 'make', 'manufacturer', 'brand'])
    year         = get(['year', 'model_year', 'reg_year', 'registration_year', 'year_of_manufacture'])
    kms          = get(['kms_driven', 'odometer', 'mileage', 'miles', 'km', 'kilometers'])
    fuel         = get(['fuel_type', 'fuel', 'fueltype'])
    transmission = get(['transmission', 'gearbox', 'gear'])
    price        = get(['price', 'selling_price', 'price_usd', 'mmr', 'saleprice', 'price_in_rupees'])

    # --- Clean price -> USD ---
    price_clean = (
        price.astype(str)
             .str.replace(r'[^\d.]', '', regex=True)
             .replace('', float('nan'))
             .astype(float)
    )
    rate = EXCHANGE_TO_USD.get(currency, 1.0)
    price_usd = (price_clean * rate).round(2)

    # --- Clean mileage -> km ---
    km_clean = (
        kms.astype(str)
           .str.replace(r'[^\d.]', '', regex=True)
           .replace('', float('nan'))
           .astype(float)
    )
    if mileage_unit == 'miles':
        km_clean = (km_clean * MILES_TO_KM).round(0)

    out = pd.DataFrame({
        'name':         name.astype(str).str.strip(),
        'company':      company.astype(str).str.strip(),
        'year':         pd.to_numeric(year, errors='coerce'),
        'kms_driven':   km_clean,
        'fuel_type':    fuel.astype(str).str.strip(),
        'transmission': transmission.astype(str).str.strip(),
        'Price_USD':    price_usd,
        'country':      country,
        'source':       'kaggle',
    })

    # Filter obviously bad rows
    out = out[
        out['year'].between(1985, 2026) &
        out['Price_USD'].between(200, 500000) &
        out['name'].str.len().gt(1)
    ]
    return out.dropna(subset=['Price_USD', 'year'])


def download_all() -> pd.DataFrame:
    os.makedirs(RAW_DIR, exist_ok=True)
    all_dfs = []

    for slug, (filename, country, currency, mileage_unit) in DATASETS.items():
        try:
            folder = _download(slug)

            if filename:
                path = os.path.join(folder, filename)
                if not os.path.exists(path):
                    # Try to find any CSV in the folder
                    csvs = [f for f in os.listdir(folder) if f.endswith('.csv')]
                    path = os.path.join(folder, csvs[0]) if csvs else None
            else:
                # UK dataset: multiple brand CSVs — merge them all
                csvs = [f for f in os.listdir(folder) if f.endswith('.csv')]
                if csvs:
                    parts = [pd.read_csv(os.path.join(folder, c)) for c in csvs]
                    df_raw = pd.concat(parts, ignore_index=True)
                    df_norm = _normalize(df_raw, country, currency, mileage_unit)
                    all_dfs.append(df_norm)
                    print(f"  {slug}: {len(df_norm)} clean rows")
                    continue
                path = None

            if path and os.path.exists(path):
                df_raw  = pd.read_csv(path, low_memory=False)
                df_norm = _normalize(df_raw, country, currency, mileage_unit)
                all_dfs.append(df_norm)
                print(f"  {slug}: {len(df_norm)} clean rows")
            else:
                print(f"  [skip] {slug}: CSV not found")

        except Exception as e:
            print(f"  [error] {slug}: {e}")

    if not all_dfs:
        print("No Kaggle datasets downloaded. Check your API key setup.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(RAW_DIR, 'kaggle_combined.csv')
    combined.to_csv(out_path, index=False)
    print(f"\nKaggle combined: {len(combined)} rows -> {out_path}")
    return combined


if __name__ == '__main__':
    download_all()
