import requests
import time
import random
import pandas as pd
from abc import ABC, abstractmethod

class BaseCarScraper(ABC):

    HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/124.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }

    # Approximate exchange rates to USD (mid-2024)
    EXCHANGE_RATES_TO_USD = {
        'USD': 1.00,
        'EUR': 1.08,
        'GBP': 1.27,
        'INR': 0.012,
        'AED': 0.272,
        'SAR': 0.267,
        'CAD': 0.735,
        'AUD': 0.655,
        'PLN': 0.250,
        'BRL': 0.195,
        'KWD': 3.250,
        'QAR': 0.275,
        'ZAR': 0.054,
        'MYR': 0.213,
        'SGD': 0.740,
    }

    def __init__(self, country: str, currency: str = 'USD', mileage_unit: str = 'km'):
        self.country = country
        self.currency = currency
        self.mileage_unit = mileage_unit          # 'km' or 'miles'
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def get(self, url: str, **kwargs):
        """Rate-limited GET request. Returns None on failure."""
        time.sleep(random.uniform(1.5, 3.5))
        try:
            resp = self.session.get(url, timeout=20, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"  [warn] {url} -> {e}")
            return None

    def to_usd(self, price: float) -> float:
        rate = self.EXCHANGE_RATES_TO_USD.get(self.currency, 1.0)
        return round(price * rate, 2)

    def to_km(self, mileage: float) -> float:
        if self.mileage_unit == 'miles':
            return round(mileage * 1.60934)
        return float(mileage)

    def standard_columns(self) -> list:
        return ['name', 'company', 'year', 'kms_driven',
                'fuel_type', 'transmission', 'Price_USD', 'country', 'source']

    def empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.standard_columns())

    @abstractmethod
    def scrape(self, pages: int = 5) -> pd.DataFrame:
        """Return a DataFrame with standard columns."""
