"""
OLX used car scraper — covers multiple countries.
OLX operates in 40+ countries with a consistent site structure.
Each country has its own domain and currency.

Supported here: UAE, Pakistan, Poland, India, South Africa, Portugal, Kenya.
"""
import json
import re
import pandas as pd
from bs4 import BeautifulSoup
from .base_scraper import BaseCarScraper

# OLX country config: code -> (base_url, country_name, currency, mileage_unit)
OLX_COUNTRIES = {
    'ae':  ('https://www.dubizzle.com',              'UAE',          'AED', 'km'),
    'pk':  ('https://www.olx.com.pk',               'Pakistan',     'PKR', 'km'),
    'pl':  ('https://www.olx.pl',                   'Poland',       'PLN', 'km'),
    'in':  ('https://www.olx.in',                   'India',        'INR', 'km'),
    'za':  ('https://www.olx.co.za',                'South Africa', 'ZAR', 'km'),
    'pt':  ('https://www.olx.pt',                   'Portugal',     'EUR', 'km'),
    'ke':  ('https://www.olx.co.ke',                'Kenya',        'KES', 'km'),
}

# PKR and KES exchange rate (approx 2024)
BaseCarScraper.EXCHANGE_RATES_TO_USD['PKR'] = 0.0036
BaseCarScraper.EXCHANGE_RATES_TO_USD['KES'] = 0.0077


class OLXScraper(BaseCarScraper):

    def __init__(self, country_codes: list = None):
        super().__init__(country='Multi', currency='USD', mileage_unit='km')
        self.country_codes = country_codes or list(OLX_COUNTRIES.keys())

    def _car_search_url(self, base: str, page: int) -> str:
        return f'{base}/motors/used-cars/?page={page}'

    def _parse_price(self, text: str) -> float | None:
        digits = re.sub(r'[^\d]', '', text)
        return float(digits) if digits else None

    def _parse_year(self, text: str) -> int | None:
        m = re.search(r'\b(19[89]\d|20[012]\d)\b', text)
        return int(m.group()) if m else None

    def _parse_mileage(self, text: str) -> int | None:
        text = text.lower().replace(',', '').replace('.', '')
        m = re.search(r'(\d+)', text)
        return int(m.group(1)) if m else None

    def _try_next_data(self, html: str) -> list:
        """Some OLX markets embed listings as __NEXT_DATA__ JSON."""
        m = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            html, re.DOTALL
        )
        if not m:
            return []
        try:
            data = json.loads(m.group(1))
            # Path varies by OLX version
            ads = (
                data.get('props', {}).get('pageProps', {}).get('ads', [])
                or data.get('props', {}).get('pageProps', {}).get('data', {}).get('ads', [])
            )
            return ads
        except Exception:
            return []

    def _parse_html_listings(self, html: str) -> list:
        """Fallback: parse OLX HTML listing cards."""
        soup = BeautifulSoup(html, 'html.parser')
        items = (
            soup.select('li[data-aut-id="itemBox"]')
            or soup.select('div.EIR5N')          # OLX India variant
            or soup.select('article.listing-card')
        )
        parsed = []
        for item in items:
            try:
                title_el = (item.select_one('[data-aut-id="itemTitle"]')
                            or item.select_one('span._2tW1I')
                            or item.select_one('h2'))
                price_el = (item.select_one('[data-aut-id="itemPrice"]')
                            or item.select_one('span._2Ks63')
                            or item.select_one('span.price'))
                title = title_el.get_text(strip=True) if title_el else ''
                price_text = price_el.get_text(strip=True) if price_el else ''
                parsed.append({'title': title, 'price_text': price_text})
            except Exception:
                continue
        return parsed

    def _scrape_country(self, code: str, base_url: str, country_name: str,
                         currency: str, mileage_unit: str, pages: int) -> pd.DataFrame:
        self.currency = currency
        self.mileage_unit = mileage_unit
        rows = []

        for page in range(1, pages + 1):
            url  = self._car_search_url(base_url, page)
            resp = self.get(url)
            if resp is None:
                continue

            # Try JSON path first
            json_ads = self._try_next_data(resp.text)
            if json_ads:
                for ad in json_ads:
                    try:
                        title = ad.get('title', '')
                        price_raw = ad.get('price', {})
                        if isinstance(price_raw, dict):
                            price = float(price_raw.get('value', 0) or 0)
                        else:
                            price = float(price_raw or 0)

                        params = {p['key']: p.get('value', '')
                                  for p in ad.get('params', []) if 'key' in p}

                        year_text = params.get('model_year', '') or params.get('year', '') or title
                        year = self._parse_year(str(year_text))

                        mileage_text = params.get('mileage', '') or params.get('milage', '')
                        mileage_raw  = self._parse_mileage(str(mileage_text)) if mileage_text else None
                        mileage_km   = self.to_km(mileage_raw) if mileage_raw else None

                        fuel  = params.get('fuel_type', 'Petrol')
                        trans = params.get('transmission', 'Unknown')
                        company = title.split()[0] if title else 'Unknown'

                        price_usd = self.to_usd(price) if price else None

                        if title and year and price_usd and 1990 <= year <= 2026:
                            rows.append({
                                'name':         title,
                                'company':      company,
                                'year':         year,
                                'kms_driven':   mileage_km,
                                'fuel_type':    fuel,
                                'transmission': trans,
                                'Price_USD':    price_usd,
                                'country':      country_name,
                                'source':       f'olx_{code}',
                            })
                    except Exception:
                        continue

            else:
                # HTML fallback
                items = self._parse_html_listings(resp.text)
                for item in items:
                    try:
                        title = item['title']
                        price = self._parse_price(item['price_text'])
                        year  = self._parse_year(title)
                        price_usd = self.to_usd(price) if price else None
                        company   = title.split()[0] if title else 'Unknown'
                        if title and year and price_usd and 1990 <= year <= 2026:
                            rows.append({
                                'name':         title,
                                'company':      company,
                                'year':         year,
                                'kms_driven':   None,
                                'fuel_type':    'Petrol',
                                'transmission': 'Unknown',
                                'Price_USD':    price_usd,
                                'country':      country_name,
                                'source':       f'olx_{code}',
                            })
                    except Exception:
                        continue

            print(f"  {country_name} page {page}/{pages}: {len(rows)} rows so far")

        return pd.DataFrame(rows, columns=self.standard_columns()) if rows else self.empty_df()

    def scrape(self, pages: int = 5) -> pd.DataFrame:
        all_dfs = []
        for code in self.country_codes:
            base_url, country_name, currency, mileage_unit = OLX_COUNTRIES[code]
            print(f"Scraping OLX: {country_name}")
            df = self._scrape_country(code, base_url, country_name, currency, mileage_unit, pages)
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else self.empty_df()
        print(f"OLX total: {len(combined)} records")
        return combined
