"""
Run all bank scrapers.
"""

import argparse
import os
import sys
import time
from multiprocessing import Process

from dotenv import load_dotenv

from ameriabank_scraper import AmeriabankScraper
from mellatbank_scraper import MellatBankScraper
from idbank_scraper import IDBankScraper

SCRAPERS = {
    "ameriabank": AmeriabankScraper,
    "mellatbank": MellatBankScraper,
    "idbank": IDBankScraper,
}

CATEGORIES = ["loans", "deposits", "branches"]


def run_bank(bank_key: str, scraper_class, category: str, output_dir: str):
    """Scrape a single bank"""
    scraper = scraper_class()

    if category == "all":
        scraper.scrape_all(output_dir)
    elif category == "loans":
        scraper.scrape_loans(output_dir)
    elif category == "deposits":
        scraper.scrape_deposits(output_dir)
    elif category == "branches":
        scraper.scrape_branches(output_dir)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Scrape Armenian bank websites")
    parser.add_argument(
        "--bank",
        choices=list(SCRAPERS.keys()) + ["all"],
        default="all",
        help="Which bank to scrape (default: all)"
    )
    parser.add_argument(
        "--category",
        choices=CATEGORIES + ["all"],
        default="all",
        help="Which category to scrape (default: all)"
    )
    parser.add_argument(
        "--output",
        default=os.getenv("DATA_PATH", "data"),
        help="Output directory (default: DATA_PATH env var or 'data')"
    )
    args = parser.parse_args()

    if args.bank == "all":
        banks_to_scrape = list(SCRAPERS.keys())
    else:
        banks_to_scrape = [args.bank]

    start = time.time()

    if len(banks_to_scrape) > 1:
        # Multiple banks: run each in its own process
        processes = []
        for bank_key in banks_to_scrape:
            p = Process(
                target=run_bank,
                args=(bank_key, SCRAPERS[bank_key], args.category, args.output),
                name=f"scraper-{bank_key}",
            )
            processes.append(p)
            p.start()
            print(f"Started process for {bank_key} (PID {p.pid})")

        for p in processes:
            p.join()

        failed = [p.name for p in processes if p.exitcode != 0]
        if failed:
            print(f"\nFailed: {', '.join(failed)}")
        else:
            print(f"\nAll banks data is successfully scraped.")
    else:
        # Single bank: run directly (no overhead of spawning a process)
        bank_key = banks_to_scrape[0]
        run_bank(bank_key, SCRAPERS[bank_key], args.category, args.output)

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()