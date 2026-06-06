"""
Data standardizer — merges all scraped + Kaggle CSVs into one clean global dataset.

Input:  data/raw/*.csv  (any source)
Output: data/processed/global_cars.csv
"""
import os
import glob
import pandas as pd
import numpy as np

RAW_DIR       = os.path.join('data', 'raw')
PROCESSED_DIR = os.path.join('data', 'processed')
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, 'global_cars.csv')

STANDARD_COLS = [
    'name', 'company', 'year', 'kms_driven',
    'fuel_type', 'transmission', 'Price_USD', 'country', 'source'
]

# Canonical fuel type mapping
FUEL_MAP = {
    'petrol': 'Petrol', 'gasoline': 'Petrol', 'gas': 'Petrol',
    'benzin': 'Petrol', 'essence': 'Petrol', 'benzine': 'Petrol',
    'diesel': 'Diesel', 'gasoil': 'Diesel',
    'electric': 'Electric', 'ev': 'Electric', 'elektro': 'Electric',
    'hybrid': 'Hybrid', 'plug-in hybrid': 'Hybrid', 'phev': 'Hybrid',
    'lpg': 'LPG', 'cng': 'CNG', 'cng/lpg': 'CNG',
    'nan': 'Petrol', 'none': 'Petrol', '': 'Petrol',
}

TRANSMISSION_MAP = {
    'automatic': 'Automatic', 'auto': 'Automatic', 'automatik': 'Automatic',
    'manual': 'Manual', 'manuell': 'Manual',
    'cvt': 'CVT', 'dsg': 'Automatic', 'pdk': 'Automatic',
    'semi-automatic': 'Semi-Auto', 'semi-auto': 'Semi-Auto',
    'unknown': 'Unknown', 'nan': 'Unknown', 'none': 'Unknown', '': 'Unknown',
}

# Known major brands (used to clean garbled company names)
KNOWN_BRANDS = {
    'audi', 'bmw', 'mercedes', 'mercedes-benz', 'volkswagen', 'vw',
    'ford', 'toyota', 'honda', 'hyundai', 'kia', 'nissan', 'mazda',
    'subaru', 'chevrolet', 'gmc', 'jeep', 'dodge', 'chrysler',
    'tesla', 'volvo', 'peugeot', 'renault', 'citroen', 'fiat',
    'seat', 'skoda', 'opel', 'vauxhall', 'land rover', 'jaguar',
    'porsche', 'lamborghini', 'ferrari', 'maserati', 'alfa romeo',
    'maruti', 'mahindra', 'tata', 'hero', 'suzuki', 'mitsubishi',
    'lexus', 'infiniti', 'acura', 'genesis', 'lincoln', 'cadillac',
    'buick', 'ram', 'rivian', 'lucid', 'geely', 'byd', 'nio', 'great wall',
}


def _normalize_fuel(val: str) -> str:
    return FUEL_MAP.get(str(val).lower().strip(), 'Petrol')


def _normalize_transmission(val: str) -> str:
    return TRANSMISSION_MAP.get(str(val).lower().strip(), 'Unknown')


def _clean_company(name: str, company: str) -> str:
    """Try to extract a clean brand from the car name if company is missing/bad."""
    company_clean = str(company).strip().lower()
    if company_clean in KNOWN_BRANDS:
        return str(company).strip().title()
    # Try first word of the car name
    first_word = str(name).split()[0].lower() if name else ''
    if first_word in KNOWN_BRANDS:
        return first_word.title()
    return str(company).strip().title() or 'Unknown'


def load_original_india() -> pd.DataFrame:
    """Load the original cleaned_car.csv (India, prices in INR)."""
    path = 'cleaned_car.csv'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={'Price': 'Price_INR'})
    df['Price_USD']    = (df['Price_INR'] * 0.012).round(2)
    df['country']      = 'India'
    df['source']       = 'quikr_original'
    df['transmission'] = 'Unknown'
    # keep standard columns
    df = df[['name', 'company', 'year', 'kms_driven', 'fuel_type',
             'transmission', 'Price_USD', 'country', 'source']]
    return df


def load_raw_csvs() -> pd.DataFrame:
    """Load all CSVs from data/raw/ that already have standard columns."""
    dfs = []
    pattern = os.path.join(RAW_DIR, '**', '*.csv')
    for path in glob.glob(pattern, recursive=True):
        try:
            df = pd.read_csv(path, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            # Accept if it has at least name + Price_USD + year
            has_std = all(c in df.columns for c in ['name', 'Price_USD', 'year'])
            if has_std:
                for col in STANDARD_COLS:
                    if col not in df.columns:
                        df[col] = None
                dfs.append(df[STANDARD_COLS])
                print(f"  Loaded {os.path.basename(path)}: {len(df)} rows")
        except Exception as e:
            print(f"  [skip] {path}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numeric coercion
    df['year']       = pd.to_numeric(df['year'],       errors='coerce')
    df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
    df['Price_USD']  = pd.to_numeric(df['Price_USD'],  errors='coerce')

    # Year sanity
    df = df[df['year'].between(1985, 2026)]

    # Price sanity (USD 300 – 500,000)
    df = df[df['Price_USD'].between(300, 500_000)]

    # Mileage cap at 700,000 km
    df['kms_driven'] = df['kms_driven'].clip(upper=700_000)

    # Normalize categoricals
    df['fuel_type']    = df['fuel_type'].apply(_normalize_fuel)
    df['transmission'] = df['transmission'].apply(_normalize_transmission)

    # Clean company names
    df['company'] = df.apply(
        lambda r: _clean_company(r['name'], r['company']), axis=1
    )

    # Drop rows missing critical fields
    df = df.dropna(subset=['name', 'year', 'Price_USD'])
    df = df[df['name'].str.strip().str.len() > 1]

    # Remove duplicates
    df = df.drop_duplicates(subset=['name', 'year', 'kms_driven', 'Price_USD', 'country'])

    df = df.reset_index(drop=True)
    return df


def merge_and_save() -> pd.DataFrame:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    parts = []

    # 1. Original India dataset
    india_df = load_original_india()
    if not india_df.empty:
        parts.append(india_df)
        print(f"Original India dataset: {len(india_df)} rows")

    # 2. All raw CSVs (from scrapers + Kaggle)
    raw_df = load_raw_csvs()
    if not raw_df.empty:
        parts.append(raw_df)

    if not parts:
        print("No data found. Run scrapers or Kaggle downloader first.")
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    print(f"\nRaw combined: {len(combined)} rows")

    cleaned = clean(combined)
    print(f"After cleaning: {len(cleaned)} rows")

    # Country breakdown
    print("\nCountry breakdown:")
    print(cleaned['country'].value_counts().to_string())

    cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved -> {OUTPUT_FILE}")
    return cleaned


if __name__ == '__main__':
    merge_and_save()
