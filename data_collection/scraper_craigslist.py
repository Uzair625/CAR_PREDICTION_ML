"""
Craigslist (USA) used car scraper — updated for 2024 site structure.
Listing container: li.cl-static-search-result
Price:  div.price  |  Title: div.title or li[title] attribute
Year extracted from title text.
"""
import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from .base_scraper import BaseCarScraper

US_CITIES = [
    'sfbay', 'losangeles', 'newyork', 'chicago', 'dallas',
    'houston', 'phoenix', 'seattle', 'denver', 'atlanta',
    'miami', 'boston', 'detroit', 'minneapolis', 'portland',
]

KNOWN_BRANDS = [
    'Toyota','Honda','Ford','Chevrolet','Chevy','BMW','Mercedes','Audi','Hyundai',
    'Kia','Nissan','Mazda','Subaru','Jeep','Dodge','Ram','GMC','Buick','Cadillac',
    'Lincoln','Chrysler','Volvo','Volkswagen','VW','Lexus','Acura','Infiniti',
    'Genesis','Tesla','Rivian','Lucid','Mitsubishi','Suzuki','Land','Mini','Fiat',
    'Porsche','Lamborghini','Ferrari','Bentley','Rolls','Maserati','Alfa',
]


class CraigslistScraper(BaseCarScraper):

    def __init__(self, cities: list = None):
        super().__init__(country='USA', currency='USD', mileage_unit='miles')
        self.cities = cities or US_CITIES[:5]

    def _search_url(self, city: str, start: int) -> str:
        return (
            f'https://{city}.craigslist.org/search/cto'
            f'?s={start}&sort=date&min_price=500&max_price=200000'
        )

    def _parse_year(self, text: str):
        m = re.search(r'\b(19[89]\d|20[012]\d)\b', text)
        return int(m.group()) if m else None

    def _parse_price(self, text: str):
        digits = re.sub(r'[^\d]', '', text)
        return int(digits) if digits else None

    def _guess_brand(self, title: str) -> str:
        words = title.split()
        for w in words:
            for brand in KNOWN_BRANDS:
                if w.lower() == brand.lower():
                    return brand
        return words[0] if words else 'Unknown'

    def _scrape_city(self, city: str, pages: int) -> list:
        rows = []
        for page in range(pages):
            start = page * 120
            url   = self._search_url(city, start)
            resp  = self.get(url)
            if resp is None:
                continue

            soup  = BeautifulSoup(resp.text, 'html.parser')
            items = soup.select('li.cl-static-search-result')

            if not items:
                print(f"  [{city}] page {page+1}: no items — skipping")
                break

            for item in items:
                try:
                    title     = item.get('title', '') or item.select_one('div.title').get_text(strip=True)
                    price_el  = item.select_one('div.price')
                    price_txt = price_el.get_text(strip=True) if price_el else ''

                    price = self._parse_price(price_txt)
                    year  = self._parse_year(title)
                    if not price or not year or not (1990 <= year <= 2026):
                        continue

                    company = self._guess_brand(title)
                    rows.append({
                        'name':         title,
                        'company':      company,
                        'year':         year,
                        'kms_driven':   None,   # not on search page; would need detail page
                        'fuel_type':    'Petrol',
                        'transmission': 'Unknown',
                        'Price_USD':    self.to_usd(price),
                        'country':      'USA',
                        'source':       f'craigslist_{city}',
                    })
                except Exception:
                    continue

            print(f"  [{city}] page {page+1}/{pages}: {len(rows)} rows")

        return rows

    def scrape(self, pages: int = 3) -> pd.DataFrame:
        all_rows = []
        for city in self.cities:
            print(f"Craigslist: {city}")
            all_rows.extend(self._scrape_city(city, pages))

        df = pd.DataFrame(all_rows, columns=self.standard_columns()) if all_rows else self.empty_df()
        print(f"Craigslist total: {len(df)} records")
        return df
