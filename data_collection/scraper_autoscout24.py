"""
AutoScout24 (Europe) scraper — updated for confirmed 2024 JSON structure.

Data path:  props.pageProps.listings[]
  listing['vehicle']['make']           → brand
  listing['vehicle']['model']          → model
  listing['vehicle']['transmission']   → Automatic/Manual
  listing['vehicle']['fuel']           → d/b/e/h  (diesel/petrol/electric/hybrid)
  listing['vehicle']['mileageInKm']    → "278,000 km"
  listing['tracking']['firstRegistration'] → "02-2014"
  listing['tracking']['price']         → "3600"
"""
import json
import re
import pandas as pd
from .base_scraper import BaseCarScraper

AS24_COUNTRIES = {
    'D':  'Germany',
    'F':  'France',
    'I':  'Italy',
    'E':  'Spain',
    'NL': 'Netherlands',
    'B':  'Belgium',
    'A':  'Austria',
    'CH': 'Switzerland',
}

FUEL_CODES = {
    'd': 'Diesel', 'b': 'Petrol', 'e': 'Electric',
    'h': 'Hybrid', 'l': 'LPG',   'c': 'CNG',
    'lpg': 'LPG',  'cng': 'CNG',
}

BASE_URL = 'https://www.autoscout24.com/lst'


class AutoScout24Scraper(BaseCarScraper):

    def __init__(self, country_codes: list = None):
        super().__init__(country='Europe', currency='EUR', mileage_unit='km')
        self.country_codes = country_codes or list(AS24_COUNTRIES.keys())

    def _url(self, code: str, page: int) -> str:
        return (
            f'{BASE_URL}?sort=standard&desc=0&ustate=N%2CU'
            f'&size=20&page={page}&cy={code}&atype=C'
        )

    def _parse_listing(self, lst: dict, country_name: str) -> dict | None:
        try:
            v = lst.get('vehicle', {})
            t = lst.get('tracking', {})

            make  = v.get('make', '')
            model = v.get('model', '')
            name  = f"{make} {model}".strip()
            if not name:
                return None

            # Year from tracking.firstRegistration e.g. "02-2014"
            reg   = t.get('firstRegistration', '')
            year_m = re.search(r'(19[89]\d|20[012]\d)', reg)
            year   = int(year_m.group()) if year_m else None
            if not year:
                return None

            # Mileage: prefer tracking (plain number), fallback to vehicle string
            km_raw  = t.get('mileage') or re.sub(r'[^\d]', '', v.get('mileageInKm', ''))
            kms     = int(km_raw) if km_raw else None

            # Price from tracking (no formatting)
            price_raw = t.get('price')
            price_eur  = float(price_raw) if price_raw else None
            if not price_eur:
                return None
            price_usd = self.to_usd(price_eur)

            # Fuel
            fuel_code = v.get('fuel', 'b').lower()
            fuel = FUEL_CODES.get(fuel_code, fuel_code.capitalize() or 'Petrol')

            # Transmission
            trans = v.get('transmission', 'Unknown') or 'Unknown'

            if not (1990 <= year <= 2026) or price_usd < 300:
                return None

            return {
                'name':         name,
                'company':      make,
                'year':         year,
                'kms_driven':   kms,
                'fuel_type':    fuel,
                'transmission': trans,
                'Price_USD':    round(price_usd, 2),
                'country':      country_name,
                'source':       'autoscout24',
            }
        except Exception:
            return None

    def _scrape_country(self, code: str, country_name: str, pages: int) -> list:
        rows = []
        for page in range(1, pages + 1):
            resp = self.get(self._url(code, page))
            if resp is None:
                continue

            m = re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                resp.text, re.DOTALL
            )
            if not m:
                print(f"  [{country_name}] page {page}: no __NEXT_DATA__")
                break

            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                break

            listings = data.get('props', {}).get('pageProps', {}).get('listings', [])
            if not listings:
                print(f"  [{country_name}] page {page}: empty listings")
                break

            for lst in listings:
                parsed = self._parse_listing(lst, country_name)
                if parsed:
                    rows.append(parsed)

            print(f"  [{country_name}] page {page}/{pages}: {len(rows)} rows")

        return rows

    def scrape(self, pages: int = 5) -> pd.DataFrame:
        all_rows = []
        for code in self.country_codes:
            country_name = AS24_COUNTRIES[code]
            print(f"AutoScout24: {country_name}")
            all_rows.extend(self._scrape_country(code, country_name, pages))

        df = pd.DataFrame(all_rows, columns=self.standard_columns()) if all_rows else self.empty_df()
        print(f"AutoScout24 total: {len(df)} records")
        return df
