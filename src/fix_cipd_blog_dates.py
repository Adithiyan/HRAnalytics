"""
fix_cipd_blog_dates.py — Add date/year to existing CIPD community blog articles.

Loads cipd_articles_static_scrape.json (604 articles, no dates),
fetches the date for each URL using the <time datetime="..."> tag,
then writes out only the 2025 articles to output/cipd_blog_2025_<timestamp>.json.

Run:
    cd E:\\Adi\\Work\\RA\\Yao\\main
    python src/fix_cipd_blog_dates.py
"""

import os
import sys
import json
import time
import random
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from config import PROJECT_ROOT, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_FILE  = os.path.join(PROJECT_ROOT, "cipd_articles_static_scrape.json")
TARGET_YEAR = 2025
MAX_WORKERS = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch_date(url):
    """Return (url, date_str, year) by fetching the page and reading <time datetime>."""
    try:
        time.sleep(random.uniform(0.3, 0.8))
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        tag = soup.find("time", attrs={"datetime": True})
        if tag:
            dt_str = tag["datetime"]          # e.g. "2025-03-15T10:00:00.000Z"
            year = int(dt_str[:4])
            date_text = tag.get_text(strip=True)  # e.g. "15 Mar 2025"
            return url, date_text, year

        # Fallback: div.content-date text
        div = soup.select_one("div.content-date")
        if div:
            text = div.get_text(strip=True)
            # Parse "22 Jun 2021" style
            try:
                dt = datetime.strptime(text, "%d %b %Y")
                return url, text, dt.year
            except Exception:
                pass

        return url, None, None
    except Exception as e:
        logging.warning(f"Date fetch failed for {url}: {e}")
        return url, None, None


def main():
    logging.info("Loading existing blog articles...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    articles = data.get("articles", [])
    logging.info(f"Loaded {len(articles)} articles")

    # Build lookup by URL
    by_url = {a["url"]: a for a in articles}
    urls = list(by_url.keys())

    logging.info(f"Fetching dates for {len(urls)} URLs ({MAX_WORKERS} workers)...")
    dated = {}
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(fetch_date, u): u for u in urls}
        for future in as_completed(future_to_url):
            url, date_text, year = future.result()
            done += 1
            by_url[url]["date"]  = date_text
            by_url[url]["year"]  = year
            if year:
                dated[url] = year
            if done % 50 == 0:
                logging.info(f"  {done}/{len(urls)} done, {len(dated)} dated so far")

    logging.info(f"Dates found for {len(dated)}/{len(urls)} articles")

    articles_2025 = [a for a in by_url.values() if a.get("year") == TARGET_YEAR]
    logging.info(f"2025 articles: {len(articles_2025)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(OUTPUT_DIR, f"cipd_blog_{TARGET_YEAR}_{timestamp}.json")
    result = {
        "source": "CIPD Community Blog",
        "year": TARGET_YEAR,
        "total_articles": len(articles_2025),
        "generated": timestamp,
        "articles": sorted(articles_2025, key=lambda a: a.get("date", ""), reverse=True),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved {len(articles_2025)} CIPD blog 2025 articles -> {out_path}")


if __name__ == "__main__":
    main()
