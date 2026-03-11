"""
phase4_shrm_articles.py — SHRM article content extraction with checkpoint/resume.

Merges links from Phase 3 output with the existing Jan 2025 links file,
deduplicates, extracts full content (5 parallel workers), filters to
TARGET_YEAR, and saves the result.

Checkpoint: src/checkpoints/shrm_articles.json
  - { url: article_dict } for every successfully processed URL.
  - Saved after every article.
  - On restart, already-processed URLs are skipped.

Intermediate save: output/shrm_2025_partial.json — updated every 25 articles.
Final output:      output/shrm_2025_<timestamp>.json

Run:
    cd E:\\Adi\\Work\\RA\\Yao\\main
    python src/phase4_shrm_articles.py

    Before running, set PHASE3_LINKS_FILE below to the file written by Phase 3,
    or leave as None to auto-detect the most recent file.
"""

import os
import sys
import json
import glob
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from config import OUTPUT_DIR
from shrm.article_extractor import extract_article_content, save_to_json
from shrm.coverage_validator import (
    build_coverage_report,
    build_static_parity_evidence,
    resolve_default_parity_paths,
)
from config import SHRM_OUTPUT_COVERAGE_PATTERN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_YEAR     = 2025
CHECKPOINT_FILE = os.path.join(SRC_DIR, "checkpoints", "shrm_articles.json")
PARTIAL_OUTPUT  = os.path.join(OUTPUT_DIR, "shrm_2025_partial.json")
SAVE_EVERY      = 25
MAX_WORKERS     = 5

# Existing Jan 2025 links file — merged with Phase 3 output
EXISTING_LINKS_FILE = os.path.join(
    OUTPUT_DIR, "shrm_article_links_news20250119_024823.txt"
)

# Set to the Phase 3 output file path, or None to auto-detect
PHASE3_LINKS_FILE = None

os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_phase3_output():
    """Return the most recent shrm_links_2025run_*.txt file, or None."""
    pattern = os.path.join(OUTPUT_DIR, "shrm_links_2025run_*.txt")
    files   = sorted(glob.glob(pattern), reverse=True)
    return files[0] if files else None


def load_links():
    """Load and deduplicate links from Phase 3 output + existing Jan 2025 file."""
    all_links = []
    seen      = set()

    def add_file(path, label):
        if not path or not os.path.exists(path):
            logging.warning(f"{label} not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        new = [l for l in lines if l not in seen]
        seen.update(new)
        all_links.extend(new)
        logging.info(f"Loaded {len(new)} new links from {label} ({path})")

    phase3_file = PHASE3_LINKS_FILE or find_phase3_output()
    add_file(phase3_file,          "Phase 3 output")
    add_file(EXISTING_LINKS_FILE,  "Jan 2025 file")

    logging.info(f"Total unique links to process: {len(all_links)}")
    return all_links


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"[Checkpoint] Loaded {len(data)} already-processed URLs.")
        return data
    return {}


def _serialize_article(article):
    """Return a copy of article with datetime_obj converted to ISO string."""
    if not isinstance(article, dict):
        return article
    a = dict(article)
    if isinstance(a.get("datetime_obj"), datetime):
        a["datetime_obj"] = a["datetime_obj"].isoformat()
    return a


def save_checkpoint(data):
    serializable = {url: _serialize_article(a) for url, a in data.items()}
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)


def save_partial(articles_2025):
    from datetime import datetime as _dt
    result = {
        "source": "SHRM", "year": TARGET_YEAR,
        "total_articles": len(articles_2025),
        "last_updated": _dt.now().isoformat(),
        "articles": [_serialize_article(a) for a in articles_2025],
    }
    with open(PARTIAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logging.info(f"[Partial] {len(articles_2025)} articles -> {PARTIAL_OUTPUT}")


def main():
    logging.info("=" * 60)
    logging.info("  Phase 4 — SHRM Article Extraction")
    logging.info(f"  Target year: {TARGET_YEAR}")
    logging.info("=" * 60)

    all_links  = load_links()
    checkpoint = load_checkpoint()

    remaining = [u for u in all_links if u not in checkpoint]
    logging.info(f"Already done: {len(checkpoint)} | Remaining: {len(remaining)}")

    # Seed 2025 list from checkpoint
    articles_2025 = [
        a for a in checkpoint.values()
        if isinstance(a, dict) and a.get("year") == TARGET_YEAR
    ]
    logging.info(f"2025 articles already in checkpoint: {len(articles_2025)}")

    processed_count = 0

    if remaining:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {
                executor.submit(extract_article_content, url): url
                for url in remaining
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    article = future.result()
                    checkpoint[url] = article
                    processed_count += 1

                    if article.get("year") == TARGET_YEAR:
                        articles_2025.append(article)
                        logging.info(
                            f"[{processed_count}/{len(remaining)}] [2025] "
                            f"{article.get('title','')[:60]}"
                        )
                    else:
                        logging.info(
                            f"[{processed_count}/{len(remaining)}] [skip] year "
                            f"{article.get('year','?')}: {url[:60]}"
                        )

                    # Save checkpoint after every article
                    save_checkpoint(checkpoint)

                    # Periodic partial save
                    if processed_count % SAVE_EVERY == 0:
                        save_partial(articles_2025)

                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")

    # Sort and write final output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(OUTPUT_DIR, f"shrm_{TARGET_YEAR}_{timestamp}.json")

    # Serialize and sort by datetime_obj
    articles_2025 = [_serialize_article(a) for a in articles_2025]

    def sort_key(a):
        raw = a.get("datetime_obj", "")
        try:
            return datetime.fromisoformat(raw) if raw else datetime.min
        except Exception:
            return datetime.min

    articles_2025.sort(key=sort_key, reverse=True)

    final_data = {
        "source": "SHRM", "year": TARGET_YEAR,
        "total_articles": len(articles_2025),
        "generated": timestamp,
        "articles": articles_2025,
    }
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    # Write SHRM parity/coverage report.
    parity_paths = resolve_default_parity_paths(os.path.dirname(SRC_DIR))
    static_evidence = build_static_parity_evidence(**parity_paths)
    coverage_report = build_coverage_report(
        target_year=TARGET_YEAR,
        links_count=len(all_links),
        articles_2025_count=len(articles_2025),
        static_evidence=static_evidence,
    )
    coverage_report["generated"] = timestamp
    coverage_path = SHRM_OUTPUT_COVERAGE_PATTERN.format(timestamp=timestamp)
    with open(coverage_path, "w", encoding="utf-8") as f:
        json.dump(coverage_report, f, indent=2, ensure_ascii=False)

    logging.info(f"\n{'='*60}")
    logging.info(f"Phase 4 complete.")
    logging.info(f"  Total URLs processed : {len(checkpoint)}")
    logging.info(f"  {TARGET_YEAR} articles found : {len(articles_2025)}")
    logging.info(f"  Final output         : {final_path}")
    logging.info(f"  Coverage report      : {coverage_path}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
