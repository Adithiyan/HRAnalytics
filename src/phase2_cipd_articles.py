"""
phase2_cipd_articles.py — CIPD article content extraction with checkpoint/resume.

Reads article URLs from cipd/scraped_links_views_uk.json (Phase 1 output),
extracts full content from each using Playwright, filters to TARGET_YEAR,
and saves the result.

Checkpoint: src/checkpoints/cipd_articles.json
  - { url: article_dict } for every successfully processed URL.
  - Saved after every article.
  - On restart, already-processed URLs are skipped.

Intermediate save: output/cipd_2025_partial.json — updated every 25 articles.
Final output:      output/cipd_2025_<timestamp>.json

Run:
    cd E:\\Adi\\Work\\RA\\Yao\\main
    python src/phase2_cipd_articles.py
"""

import os
import sys
import json
import random
import time
from datetime import datetime
from playwright.sync_api import sync_playwright

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from config import (
    OUTPUT_DIR, CIPD_ARTICLE_LINKS_FILE, CIPD_STORAGE_STATE, CIPD_HOME_URL,
)
from cipd.article_extractor import extract_article_content, handle_cookie_banner, login, parse_date

TARGET_YEAR      = 2025
CHECKPOINT_FILE  = os.path.join(SRC_DIR, "checkpoints", "cipd_articles.json")
PARTIAL_OUTPUT   = os.path.join(OUTPUT_DIR, "cipd_2025_partial.json")
SAVE_EVERY       = 25

os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Checkpoint] Loaded {len(data)} already-processed URLs.")
        return data
    return {}


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def save_partial(articles_2025, target_year=TARGET_YEAR):
    result = {
        "source": "CIPD", "year": target_year,
        "total_articles": len(articles_2025),
        "last_updated": datetime.now().isoformat(),
        "articles": sorted(articles_2025, key=parse_date, reverse=True),
    }
    with open(PARTIAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [Partial] Saved {len(articles_2025)} articles -> {PARTIAL_OUTPUT}")


def main(target_year=TARGET_YEAR, final_output_path=None):
    print("=" * 60)
    print("  Phase 2 — CIPD Article Extraction")
    print(f"  Target year: {target_year}")
    print("=" * 60)

    # Load article URL list from Phase 1 output
    with open(CIPD_ARTICLE_LINKS_FILE, "r") as f:
        scraped_data = json.load(f)

    all_urls = []
    for cat_data in scraped_data.values():
        all_urls.extend(cat_data.get("articles", []))
    all_urls = list(dict.fromkeys(all_urls))  # deduplicate, preserve order
    print(f"Total unique article URLs: {len(all_urls)}")

    # Load checkpoint
    checkpoint = load_checkpoint()
    remaining_urls = [u for u in all_urls if u not in checkpoint]
    print(f"Already done: {len(checkpoint)} | Remaining: {len(remaining_urls)}")

    # Collect 2025 articles from checkpoint data
    articles_2025 = [
        a for a in checkpoint.values()
        if isinstance(a, dict) and a.get("year") == target_year
    ]
    print(f"{target_year} articles already in checkpoint: {len(articles_2025)}")

    if not remaining_urls:
        print("All URLs already processed.")
    else:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = (
                browser.new_context(storage_state=CIPD_STORAGE_STATE)
                if os.path.exists(CIPD_STORAGE_STATE)
                else browser.new_context()
            )
            page = context.new_page()

            try:
                page.goto(CIPD_HOME_URL)
                page.wait_for_load_state("networkidle")
                handle_cookie_banner(page)
                context.storage_state(path=CIPD_STORAGE_STATE)

                for i, url in enumerate(remaining_urls, 1):
                    print(f"\n[{i}/{len(remaining_urls)}] {url}")
                    article = extract_article_content(page, url)
                    checkpoint[url] = article

                    if article.get("year") == target_year:
                        articles_2025.append(article)
                        print(f"  [{target_year}] {article.get('title','')[:60]}")
                    else:
                        print(f"  [skip] Year {article.get('year','?')}")

                    # Save checkpoint after every article
                    save_checkpoint(checkpoint)

                    # Periodic session save + partial output
                    if i % SAVE_EVERY == 0:
                        context.storage_state(path=CIPD_STORAGE_STATE)
                        save_partial(articles_2025, target_year=target_year)
                        print(f"  Progress: {i}/{len(remaining_urls)} processed, "
                              f"{len(articles_2025)} from {target_year}")

            finally:
                context.storage_state(path=CIPD_STORAGE_STATE)
                browser.close()

    # Write final output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = final_output_path or os.path.join(OUTPUT_DIR, f"cipd_{target_year}_{timestamp}.json")
    final_data = {
        "source": "CIPD", "year": target_year,
        "total_articles": len(articles_2025),
        "generated": timestamp,
        "articles": sorted(articles_2025, key=parse_date, reverse=True),
    }
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Phase 2 complete.")
    print(f"  Total URLs processed : {len(checkpoint)}")
    print(f"  {target_year} articles found : {len(articles_2025)}")
    print(f"  Final output         : {final_path}")
    print(f"{'='*60}")
    return {
        "total_urls_processed": len(checkpoint),
        "total_articles_for_year": len(articles_2025),
        "output_file": final_path,
    }


if __name__ == "__main__":
    main()
