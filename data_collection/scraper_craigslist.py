"""
Craigslist (USA) used car scraper.
Targets the 'cars+trucks by owner' and 'by dealer' sections across major US cities.
Craigslist is publicly accessible with no login required.
"""
import re
import pandas as pd
from bs4 import BeautifulSoup
from .base_scraper import BaseCarScraper

# Top US cities with high car listing volume
US_CITIES = [
    'losangeles', 'sfbay', 'newyork', 'chicago', 'dallas',
    'houston', 'phoenix', 'seattle', 'denver', 'atlanta',
    'miami', 'boston', 'detroit', 'minneapolis', 'portland',
]

SECTION = 'cto'   # cars+trucks by owner  (use 'ctd' for dealers)


class CraigslistScraper(BaseCarScraper):

    def __init__(self, cities: list = None):
        super().__init__(country='USA', currency='USD', mileage_unit='miles')
        self.cities = cities or US_CITIES[:5]   # default: 5 cities

    def _search_url(self, city: str, page: int) -> str:
        offset = page * 120
        return (
            f'https://{city}.craigslist.org/search/{SECTION}'
            f'?sort=date&min_price=500&max_price=150000'
            f'#search=1~list~{page}~0'
        )

    def _search_url_v2(self, city: str, start: int) -> str:
        """Fallback URL with start offset (older Craigslist format)."""
        return (
            f'https://{city}.craigslist.org/search/{SECTION}'
            f'?s={start}&sort=date&min_price=500&max_price=150000'
        )

    def _parse_year(self, text: str):
        match = re.search(r'\b(19[89]\d|20[012]\d)\b', text)
        return int(match.group()) if match else None

    def _parse_price(self, text: str):
        digits = re.sub(r'[^\d]', '', text)
        return int(digits) if digits else None

    def _parse_mileage(self, text: str):
        text = text.lower().replace(',', '')
        match = re.search(r'(\d+)\s*(mi|km|miles|kilometers)?', text)
        if match:
            return int(match.group(1))
        return None

    def _scrape_city(self, city: str, pages: int) -> pd.DataFrame:
        rows = []
        for page in range(pages):
            url = self._search_url_v2(city, page * 120)
            resp = self.get(url)
            if resp is None:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Current Craigslist structure (2024)
            results = soup.select('li.cl-search-result')
            if not results:
                # Fallback to older structure
                results = soup.select('li.result-row')

            if not results:
                print(f"  [info] {city} page {page}: no results found — site structure may have changed")
                break

            for item in results:
                try:
                    # Title / name
                    title_el = item.select_one('.posting-title .label') or \
                               item.select_one('.result-title') or \
                               item.select_one('a[data-id]')
                    title = title_el.get_text(strip=True) if title_el else ''

                    # Price
                    price_el = item.select_one('.priceinfo') or \
                               item.select_one('.result-price')
                    price_text = price_el.get_text(strip=True) if price_el else ''
                    price = self._parse_price(price_text)

                    # Year from title
                    year = self._parse_year(title)

                    # Attributes (odometer, condition, etc.)
                    attrs = {
                        span.get('data-attr', ''): span.get_text(strip=True)
                        for span in item.select('[data-attr]')
                    }
                    odometer_text = attrs.get('auto_miles', '') or attrs.get('auto_kilometers', '')
                    mileage_raw = self._parse_mileage(odometer_text) if odometer_text else None
                    mileage_km = self.to_km(mileage_raw) if mileage_raw else None

                    # Try to extract company (first word of title that's a known brand)
                    words = title.split()
                    company = words[0] if words else 'Unknown'

                    fuel_type = attrs.get('auto_fuel_type', 'Petrol').capitalize()
                    transmission = attrs.get('auto_transmission', 'Unknown')

                    if price and year and 1990 <= year <= 2026:
                        rows.append({
                            'name': title,
                            'company': company,
                            'year': year,
                            'kms_driven': mileage_km,
                            'fuel_type': fuel_type,
                            'transmission': transmission,
                            'Price_USD': self.to_usd(price),
                            'country': self.country,
                            'source': f'craigslist_{city}',
                        })
                except Exception as e:
                    print(f"  [parse error] {city}: {e}")
                    continue

            print(f"  {city} page {page+1}/{pages}: {len(rows)} rows so far")

        return pd.DataFrame(rows, columns=self.standard_columns())

    def scrape(self, pages: int = 3) -> pd.DataFrame:
        all_dfs = []
        for city in self.cities:
            print(f"Scraping Craigslist: {city}")
            df = self._scrape_city(city, pages)
            all_dfs.append(df)
        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else self.empty_df()
        print(f"Craigslist total: {len(combined)} records")
        return combined
