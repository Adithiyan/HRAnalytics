"""
run_2025.py — Orchestrate 2025 article collection from CIPD and SHRM.

This script chains the individual pipeline stages in order and post-filters
all extracted articles to TARGET_YEAR (2025).  Each stage writes its own
intermediate files so that you can restart from any point if a stage fails.

Pipeline order
--------------
CIPD:
  Unified  run_cipd_2025.py (knowledge + community + peoplemanagement + podcasts)
  Filter   year == 2025 (knowledge output) → output/cipd_2025_<ts>.json

SHRM:
  Stage 1  shrm/link_collector.py   → output/shrm_article_links_news<ts>.txt
  Stage 2  shrm/article_extractor.py→ output/shrm_articles_all_<ts>.json
  Filter   year == 2025             → output/shrm_2025_<ts>.json

Tuning
------
  SHRM_MAX_PAGES  — how many SHRM result pages to crawl (10 per page).
                    300 pages ≈ 3,000 most-recent articles, which covers all
                    of 2025 when run in 2026.  Reduce for faster testing.
  SKIP_CIPD / SKIP_SHRM — set to True to re-use previously collected data
                    and only re-run the filter step.

Usage:
    cd E:\\Adi\\Work\\RA\\Yao\\main
    python src/run_2025.py
"""

import os
import sys
import json
import logging
from datetime import datetime

# Make src/ importable regardless of working directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from config import (
    OUTPUT_DIR,
    CIPD_ARTICLE_DATA_FILE,
    SHRM_TOPICS_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------
TARGET_YEAR    = 2025
SHRM_MAX_PAGES = 400   # ~4,000 articles; safely covers all of 2025 + early 2026 buffer

SKIP_CIPD = False   # Set True to skip CIPD stages 1-3 and only filter existing output
SKIP_SHRM = False   # Set True to skip SHRM stages 1-2 and only filter existing output

# If SKIP_SHRM=True, point this at the full SHRM articles JSON from a prior run
SHRM_EXISTING_JSON = os.path.join(OUTPUT_DIR, "")  # fill in filename if skipping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_to_year(articles, year):
    """Return only articles whose 'year' field equals `year`."""
    return [a for a in articles if a.get("year") == year]


def save_json(data, path):
    """Write `data` as indented JSON to `path`, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved → {path}")


# ---------------------------------------------------------------------------
# CIPD pipeline
# ---------------------------------------------------------------------------

def run_cipd_2025(timestamp):
    """
    Run CIPD stages 1-3 then filter to TARGET_YEAR.

    Returns the filtered article list and writes the result to
    output/cipd_2025_<timestamp>.json.
    """
    if not SKIP_CIPD:
        logging.info("── CIPD unified 2025 pipeline: knowledge + community + peoplemanagement + podcast parity ──")
        from run_cipd_2025 import run_pipeline as run_cipd_pipeline
        summary = run_cipd_pipeline(target_year=TARGET_YEAR, include_podcast_stage=True)
        knowledge_out = summary.get("knowledge_output")
        if not knowledge_out or not os.path.exists(knowledge_out):
            raise FileNotFoundError("Unified CIPD pipeline did not produce knowledge output.")
        with open(knowledge_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_articles = data.get("articles", [])
        articles_2025 = all_articles
        logging.info(
            f"CIPD: {len(articles_2025)} articles published in {TARGET_YEAR} "
            f"(knowledge output: {knowledge_out})"
        )
        return articles_2025
    else:
        logging.info("SKIP_CIPD=True — using existing CIPD article data file.")

    # Load the full extraction output and filter
    with open(CIPD_ARTICLE_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_articles    = data.get("articles", [])
    articles_2025   = filter_to_year(all_articles, TARGET_YEAR)

    logging.info(
        f"CIPD: {len(articles_2025)} / {len(all_articles)} articles "
        f"published in {TARGET_YEAR}"
    )

    out_path = os.path.join(OUTPUT_DIR, f"cipd_{TARGET_YEAR}_{timestamp}.json")
    save_json(
        {
            "source":         "CIPD",
            "year":           TARGET_YEAR,
            "total_articles": len(articles_2025),
            "articles":       articles_2025,
        },
        out_path,
    )
    return articles_2025


# ---------------------------------------------------------------------------
# SHRM pipeline
# ---------------------------------------------------------------------------

def run_shrm_2025(timestamp):
    """
    Run SHRM stages 1-2 then filter to TARGET_YEAR.

    Stage 1 collects SHRM_MAX_PAGES pages of links (newest first).
    Stage 2 extracts content for every collected URL.
    The result is filtered to TARGET_YEAR and written to
    output/shrm_2025_<timestamp>.json.
    """
    from shrm.article_extractor import process_links, save_to_json

    if not SKIP_SHRM:
        logging.info(f"── SHRM Stage 1: collecting links (max {SHRM_MAX_PAGES} pages) ──")
        from shrm.link_collector import collect_article_links
        links = collect_article_links(SHRM_TOPICS_URL, max_pages=SHRM_MAX_PAGES)

        if not links:
            logging.error("No SHRM links collected — aborting SHRM pipeline.")
            return []

        logging.info(f"── SHRM Stage 2: extracting content for {len(links)} URLs ──")
        all_articles_file = f"shrm_articles_all_{timestamp}.json"
        process_links(links, all_articles_file)
        shrm_json_path = os.path.join(OUTPUT_DIR, all_articles_file)
    else:
        if not SHRM_EXISTING_JSON or not os.path.exists(SHRM_EXISTING_JSON):
            logging.error(
                "SKIP_SHRM=True but SHRM_EXISTING_JSON is not set or does not exist."
            )
            return []
        shrm_json_path = SHRM_EXISTING_JSON
        logging.info(f"SKIP_SHRM=True — loading existing file: {shrm_json_path}")

    with open(shrm_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_articles  = data.get("articles", [])
    articles_2025 = filter_to_year(all_articles, TARGET_YEAR)

    logging.info(
        f"SHRM: {len(articles_2025)} / {len(all_articles)} articles "
        f"published in {TARGET_YEAR}"
    )

    out_path = os.path.join(OUTPUT_DIR, f"shrm_{TARGET_YEAR}_{timestamp}.json")
    save_json(
        {
            "source":         "SHRM",
            "year":           TARGET_YEAR,
            "total_articles": len(articles_2025),
            "articles":       articles_2025,
        },
        out_path,
    )
    return articles_2025


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(f"{'='*60}")
    logging.info(f"  2025 Data Collection Run  —  {timestamp}")
    logging.info(f"{'='*60}")

    cipd_articles = run_cipd_2025(timestamp)
    shrm_articles = run_shrm_2025(timestamp)

    logging.info(f"\n{'='*60}")
    logging.info(f"  Run complete")
    logging.info(f"  CIPD {TARGET_YEAR}: {len(cipd_articles)} articles")
    logging.info(f"  SHRM {TARGET_YEAR}: {len(shrm_articles)} articles")
    logging.info(f"  Outputs in: {OUTPUT_DIR}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
