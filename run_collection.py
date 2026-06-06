"""
Master data collection runner.

Usage:
  python run_collection.py                  # run all sources
  python run_collection.py --source kaggle  # only Kaggle
  python run_collection.py --source scrape  # only live scrapers
  python run_collection.py --source craigslist autoscout24
  python run_collection.py --pages 3        # 3 pages per scraper (default: 5)

After collection, automatically standardizes + merges data, then retrains the model.
"""
import os
import sys
import argparse
import pandas as pd

RAW_DIR = os.path.join('data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)


def run_kaggle():
    print("\n=== Kaggle Datasets ===")
    from data_collection.kaggle_downloader import download_all
    return download_all()


def run_craigslist(pages: int, cities: list = None):
    print("\n=== Craigslist (USA) ===")
    from data_collection.scraper_craigslist import CraigslistScraper
    scraper = CraigslistScraper(cities=cities)
    df = scraper.scrape(pages=pages)
    if not df.empty:
        path = os.path.join(RAW_DIR, 'craigslist_usa.csv')
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows -> {path}")
    return df


def run_autoscout24(pages: int, country_codes: list = None):
    print("\n=== AutoScout24 (Europe) ===")
    from data_collection.scraper_autoscout24 import AutoScout24Scraper
    scraper = AutoScout24Scraper(country_codes=country_codes)
    df = scraper.scrape(pages=pages)
    if not df.empty:
        path = os.path.join(RAW_DIR, 'autoscout24_europe.csv')
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows -> {path}")
    return df


def run_olx(pages: int, country_codes: list = None):
    print("\n=== OLX (Multi-country) ===")
    from data_collection.scraper_olx import OLXScraper
    scraper = OLXScraper(country_codes=country_codes)
    df = scraper.scrape(pages=pages)
    if not df.empty:
        path = os.path.join(RAW_DIR, 'olx_multicountry.csv')
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows -> {path}")
    return df


def standardize():
    print("\n=== Standardizing & Merging All Sources ===")
    from data_collection.standardize import merge_and_save
    return merge_and_save()


def retrain():
    print("\n=== Retraining Model on Global Data ===")
    import train_model
    train_model.run(data_path=os.path.join('data', 'processed', 'global_cars.csv'))


def main():
    parser = argparse.ArgumentParser(description='Car price data collection pipeline')
    parser.add_argument('--source', nargs='+',
                        choices=['kaggle', 'craigslist', 'autoscout24', 'olx', 'all'],
                        default=['all'],
                        help='Data sources to collect from')
    parser.add_argument('--pages', type=int, default=5,
                        help='Number of pages to scrape per site (default: 5)')
    parser.add_argument('--no-train', action='store_true',
                        help='Skip model retraining after collection')
    args = parser.parse_args()

    sources = set(args.source)
    run_all = 'all' in sources

    collected = []

    if run_all or 'kaggle' in sources:
        try:
            df = run_kaggle()
            if df is not None and not df.empty:
                collected.append(df)
        except Exception as e:
            print(f"  [kaggle failed] {e}")
            print("  -> Make sure kaggle.json is set up. See data_collection/kaggle_downloader.py")

    if run_all or 'craigslist' in sources:
        try:
            df = run_craigslist(args.pages)
            if not df.empty:
                collected.append(df)
        except Exception as e:
            print(f"  [craigslist failed] {e}")

    if run_all or 'autoscout24' in sources:
        try:
            df = run_autoscout24(args.pages)
            if not df.empty:
                collected.append(df)
        except Exception as e:
            print(f"  [autoscout24 failed] {e}")

    if run_all or 'olx' in sources:
        try:
            df = run_olx(args.pages)
            if not df.empty:
                collected.append(df)
        except Exception as e:
            print(f"  [olx failed] {e}")

    # Always standardize + merge (includes the original India dataset)
    global_df = standardize()

    if not args.no_train and global_df is not None and not global_df.empty:
        retrain()
    elif args.no_train:
        print("\nSkipping retraining (--no-train flag set).")
    else:
        print("\nNo data collected — retraining skipped.")

    print("\nDone.")


if __name__ == '__main__':
    main()
