"""
phase1_cipd_links.py — CIPD article link collection with checkpoint/resume.

Reads the 196 category URLs already collected in cipd/main_links_views_uk.txt,
visits each with Playwright, paginates through all pages, and collects
div.card.card--full article hrefs.

Checkpoint: src/checkpoints/cipd_links.json
  - Saved after every completed category.
  - On restart, already-completed categories are skipped automatically.

Output: cipd/scraped_links_views_uk.json

Run:
    cd E:\\Adi\\Work\\RA\\Yao\\main
    python src/phase1_cipd_links.py
"""

import os
import sys
import json

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from config import CIPD_ARTICLE_LINKS_FILE, CIPD_STORAGE_STATE, CIPD_HOME_URL, PROJECT_ROOT

# Use main_links_views.txt — the 18 knowledge-hub listing pages that actually
# contain div.card.card--full article cards.  (main_links_views_uk.txt contains
# 196 navigation/homepage links with no article cards and is NOT used here.)
CIPD_MAIN_LINKS_FILE = os.path.join(PROJECT_ROOT, "cipd", "main_links_views.txt")
from cipd.article_links import scrape_category, handle_cookie_banner, login
from playwright.sync_api import sync_playwright

CHECKPOINT_FILE = os.path.join(SRC_DIR, "checkpoints", "cipd_links.json")
os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        print(f"[Checkpoint] Loaded {len(data)} completed categories.")
        return data
    return {}


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def main():
    print("=" * 60)
    print("  Phase 1 — CIPD Link Collection")
    print("=" * 60)

    with open(CIPD_MAIN_LINKS_FILE, "r") as f:
        raw = [line.strip() for line in f if line.strip()]
    # Deduplicate while preserving order
    seen_cats = set()
    all_categories = [c for c in raw if not (c in seen_cats or seen_cats.add(c))]
    print(f"Total categories: {len(all_categories)} (from {len(raw)} lines, deduplicated)")

    # Load checkpoint — already completed categories are skipped
    scraped_data = load_checkpoint()
    remaining = [c for c in all_categories if c not in scraped_data]
    print(f"Already done: {len(scraped_data)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All categories already scraped. Writing final output.")
        with open(CIPD_ARTICLE_LINKS_FILE, "w") as f:
            json.dump(scraped_data, f, indent=4)
        return {
            "categories_scraped": len(scraped_data),
            "total_articles": sum(v["total_articles"] for v in scraped_data.values()),
            "output_file": CIPD_ARTICLE_LINKS_FILE,
        }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = (
            browser.new_context(storage_state=CIPD_STORAGE_STATE)
            if os.path.exists(CIPD_STORAGE_STATE)
            else browser.new_context()
        )
        page = context.new_page()

        try:
            # Always do a fresh login to ensure full member access.
            # Without this, CIPD shows only 12 articles per page (guest view).
            print("Logging in to CIPD for full member access...")
            login(page)
            page.goto(CIPD_HOME_URL)
            page.wait_for_load_state("networkidle")
            handle_cookie_banner(page)
            context.storage_state(path=CIPD_STORAGE_STATE)

            for i, category_url in enumerate(remaining, 1):
                print(f"\n[{i}/{len(remaining)}] {category_url}")
                links = scrape_category(page, category_url)
                scraped_data[category_url] = {
                    "articles":       links,
                    "total_articles": len(links),
                }
                save_checkpoint(scraped_data)
                print(f"  Saved checkpoint. Total categories done: {len(scraped_data)}")

        finally:
            context.storage_state(path=CIPD_STORAGE_STATE)
            browser.close()

    # Write final consolidated output
    with open(CIPD_ARTICLE_LINKS_FILE, "w") as f:
        json.dump(scraped_data, f, indent=4)

    total_articles = sum(v["total_articles"] for v in scraped_data.values())
    print(f"\n{'='*60}")
    print(f"Phase 1 complete.")
    print(f"  Categories scraped : {len(scraped_data)}")
    print(f"  Total article links: {total_articles}")
    print(f"  Output             : {CIPD_ARTICLE_LINKS_FILE}")
    print(f"{'='*60}")
    return {
        "categories_scraped": len(scraped_data),
        "total_articles": total_articles,
        "output_file": CIPD_ARTICLE_LINKS_FILE,
    }


if __name__ == "__main__":
    main()
