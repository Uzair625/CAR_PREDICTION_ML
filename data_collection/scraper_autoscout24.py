"""
AutoScout24 (Europe) used car scraper.
AutoScout24 embeds all listing data as JSON inside a <script id="__NEXT_DATA__"> tag,
making it easy to parse without a headless browser.

Covers: Germany, France, Italy, Spain, Netherlands, Belgium, Austria, Switzerland.
"""
import json
import re
import pandas as pd
from .base_scraper import BaseCarScraper

# AutoScout24 country codes -> (display name, currency)
AS24_COUNTRIES = {
    'D':  ('Germany',     'EUR'),
    'F':  ('France',      'EUR'),
    'I':  ('Italy',       'EUR'),
    'E':  ('Spain',       'EUR'),
    'NL': ('Netherlands', 'EUR'),
    'B':  ('Belgium',     'EUR'),
    'A':  ('Austria',     'EUR'),
    'CH': ('Switzerland', 'EUR'),
}

BASE_URL = 'https://www.autoscout24.com/lst'


class AutoScout24Scraper(BaseCarScraper):

    def __init__(self, country_codes: list = None):
        super().__init__(country='Europe', currency='EUR', mileage_unit='km')
        self.country_codes = country_codes or list(AS24_COUNTRIES.keys())

    def _build_url(self, country_code: str, page: int) -> str:
        return (
            f'{BASE_URL}?sort=standard&desc=0'
            f'&ustate=N%2CU'          # new and used
            f'&size=20'
            f'&page={page}'
            f'&cy={country_code}'
            f'&atype=C'               # cars only
        )

    def _extract_next_data(self, html: str) -> dict:
        """Extract the __NEXT_DATA__ JSON blob from the HTML."""
        match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            html, re.DOTALL
        )
        if not match:
            return {}
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}

    def _parse_listing(self, listing: dict, country_name: str) -> dict | None:
        try:
            tracking = listing.get('tracking', {})
            attrs    = listing.get('attributes', [])
            attr_map = {a.get('id'): a.get('value') for a in attrs if a.get('id')}

            make  = tracking.get('make', '') or listing.get('make', {}).get('name', '')
            model = tracking.get('model', '') or listing.get('modelName', '')
            name  = f"{make} {model}".strip() or listing.get('title', '')

            year_raw  = tracking.get('firstRegistration', '') or attr_map.get('registrationDate', '')
            year_match = re.search(r'(19[89]\d|20[012]\d)', str(year_raw))
            year = int(year_match.group()) if year_match else None

            mileage = (
                listing.get('mileage')
                or attr_map.get('mileage')
                or tracking.get('mileage')
            )
            mileage = int(re.sub(r'[^\d]', '', str(mileage))) if mileage else None

            price_info = listing.get('prices', {}).get('public', {})
            price_eur  = price_info.get('priceRaw') or price_info.get('price')
            if price_eur is None:
                price_eur = listing.get('price')
            price_usd = self.to_usd(float(price_eur)) if price_eur else None

            fuel  = attr_map.get('fuel', tracking.get('fuel', 'Petrol'))
            trans = attr_map.get('transmissionType', tracking.get('transmission', 'Unknown'))

            if not (name and year and price_usd and 1990 <= year <= 2026):
                return None

            return {
                'name':         name,
                'company':      make or name.split()[0],
                'year':         year,
                'kms_driven':   mileage,
                'fuel_type':    fuel,
                'transmission': trans,
                'Price_USD':    price_usd,
                'country':      country_name,
                'source':       'autoscout24',
            }
        except Exception:
            return None

    def _scrape_country(self, code: str, country_name: str, pages: int) -> pd.DataFrame:
        rows = []
        for page in range(1, pages + 1):
            url  = self._build_url(code, page)
            resp = self.get(url)
            if resp is None:
                continue

            data = self._extract_next_data(resp.text)
            listings = (
                data.get('props', {})
                    .get('pageProps', {})
                    .get('listings', [])
            )

            if not listings:
                # Try alternative path in the JSON tree
                listings = (
                    data.get('props', {})
                        .get('pageProps', {})
                        .get('searchResponse', {})
                        .get('listings', [])
                )

            if not listings:
                print(f"  [info] {country_name} page {page}: no listings in JSON — structure may have changed")
                break

            for lst in listings:
                parsed = self._parse_listing(lst, country_name)
                if parsed:
                    rows.append(parsed)

            print(f"  {country_name} page {page}/{pages}: {len(rows)} rows so far")

        return pd.DataFrame(rows, columns=self.standard_columns()) if rows else self.empty_df()

    def scrape(self, pages: int = 5) -> pd.DataFrame:
        all_dfs = []
        for code in self.country_codes:
            country_name, _ = AS24_COUNTRIES[code]
            print(f"Scraping AutoScout24: {country_name}")
            df = self._scrape_country(code, country_name, pages)
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else self.empty_df()
        print(f"AutoScout24 total: {len(combined)} records")
        return combined
