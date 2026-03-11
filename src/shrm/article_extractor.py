"""
shrm/article_extractor.py — Stage 2: Extract full article content from SHRM URLs.

Reads a plain-text file of SHRM article URLs (one per line), fetches each
article using Playwright, and extracts structured metadata.  Extraction runs
concurrently (5 workers by default) and the output JSON is saved after every
article so progress survives interruptions.  Articles are sorted newest-first
in the output.

Extracted fields per article:
    url, title, category, date, year, datetime_obj, author, type, tags,
    full_text  (and optionally fallback_content / error on failure)

Input:  Plain-text file of URLs — edit `input_file` in __main__ block.
Output: output/articles_<timestamp>.json

Usage:
    python -m shrm.article_extractor      # from src/
    python src/shrm/article_extractor.py  # from project root
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from playwright.sync_api import sync_playwright
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text):
    """Strip whitespace, newlines, and SHRM boilerplate from extracted strings."""
    if not text:
        return "No Data"
    return text.replace("opens in a new tab", "").strip().replace("\n", "").replace("\r", "")


# ---------------------------------------------------------------------------
# Single-article extraction
# ---------------------------------------------------------------------------

def extract_article_content(url):
    """
    Fetch one SHRM article and return a dict of its structured content.

    Opens a headless Chromium browser, navigates to `url`, and scrapes:
      - title        — h1.content__title
      - category     — breadcrumb items (3rd level onward joined with ' > ')
      - date         — span.content__date  (parsed as "%B %d, %Y")
      - year         — integer year from date, or "Unknown"
      - datetime_obj — datetime for sorting (or datetime.min on failure)
      - author       — span.content__author
      - type         — div.pretitle a[data-contentfiltertag]
      - tags         — a[aria-label="button tag"]  (type tag excluded)
      - full_text    — paragraphs/headings/lists inside div.cmp-text

    On any field-level failure the field is set to a safe default and a
    WARNING is logged.  On a page-level failure an 'error' key is added.

    Args:
        url (str): Fully-qualified SHRM article URL.

    Returns:
        dict: Extracted article data keyed as described above.
    """
    content_data = {"url": url}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page    = browser.new_page()

            logging.info(f"Extracting content from: {url}")
            page.goto(url, timeout=10000)
            page.wait_for_load_state("domcontentloaded")

            # Title
            try:
                content_data["title"] = clean_text(
                    page.text_content('h1.content__title', timeout=5000) or "No Title"
                )
            except Exception as e:
                logging.warning(f"Title not found for {url}: {e}")
                content_data["title"] = "No Title"

            # Category (breadcrumb from 3rd crumb onward)
            try:
                breadcrumb_items = page.query_selector_all(
                    'ol.cmp-breadcrumb__list li.cmp-breadcrumb__item span[itemprop="name"]'
                )
                if len(breadcrumb_items) >= 2:
                    content_data["category"] = " > ".join(
                        clean_text(item.text_content()) for item in breadcrumb_items[2:]
                    )
                else:
                    content_data["category"] = "No Category"
            except Exception as e:
                logging.warning(f"Category not found for {url}: {e}")
                content_data["category"] = "No Category"

            # Date
            try:
                date = clean_text(
                    page.text_content('span.content__date', timeout=5000) or "No Date"
                )
                content_data["date"]         = date
                content_data["year"]         = datetime.strptime(date, "%B %d, %Y").year
                content_data["datetime_obj"] = datetime.strptime(date, "%B %d, %Y")
            except Exception as e:
                logging.warning(f"Date not found or invalid for {url}: {e}")
                content_data["date"]         = "No Date"
                content_data["year"]         = "Unknown"
                content_data["datetime_obj"] = datetime.min

            # Author
            try:
                content_data["author"] = clean_text(
                    page.text_content('span.content__author', timeout=5000) or "No Author"
                )
            except Exception as e:
                logging.warning(f"Author not found for {url}: {e}")
                content_data["author"] = "No Author"

            # Content type
            try:
                type_tag = page.query_selector('div.pretitle a[data-contentfiltertag]')
                content_data["type"] = clean_text(type_tag.text_content() if type_tag else "Unknown")
            except Exception as e:
                logging.warning(f"Type not found for {url}: {e}")
                content_data["type"] = "Unknown"

            # Tags (deduplicated against type)
            try:
                tags     = page.query_selector_all('a[aria-label="button tag"]')
                tag_list = [clean_text(tag.text_content()) for tag in tags]
                content_data["tags"] = [
                    tag for tag in tag_list
                    if tag.lower() != content_data["type"].lower()
                ]
            except Exception as e:
                logging.warning(f"Tags not found for {url}: {e}")
                content_data["tags"] = []

            # Full article text
            try:
                main_content_div = page.query_selector('div.cmp-text')
                if main_content_div:
                    paragraphs = main_content_div.query_selector_all('p, h2, ul')
                    content_data["full_text"] = "\n\n".join(
                        clean_text(p.text_content()) for p in paragraphs
                    )
                else:
                    content_data["full_text"] = clean_text(page.content()) or "No Content"
            except Exception as e:
                logging.warning(f"Full text extraction failed for {url}: {e}")
                content_data["full_text"] = clean_text(page.content()) or "No Content"

            browser.close()

    except Exception as e:
        logging.error(f"Error extracting content from {url}: {e}")
        content_data["error"] = str(e)
        try:
            content_data["fallback_content"] = clean_text(page.content())
        except Exception:
            content_data["fallback_content"] = "No Additional Content"

    return content_data


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_to_json(data, file_name):
    """
    Serialise `data` to JSON and write it to OUTPUT_DIR/<file_name>.

    The output directory is created if it does not already exist.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logging.info(f"Data saved to {file_path}")


# ---------------------------------------------------------------------------
# Concurrent processing
# ---------------------------------------------------------------------------

def process_links(links, output_file):
    """
    Extract article content for a list of URLs using a thread pool.

    Uses up to 5 concurrent Playwright browser instances.  After each
    completed future the full accumulated result set is sorted by date
    (newest first) and flushed to disk, so the output JSON always reflects
    current progress.

    Args:
        links (list[str]): Article URLs to process.
        output_file (str): Filename (not full path) written inside OUTPUT_DIR.

    Returns:
        list[dict]: All extracted article dicts (unsorted).
    """
    all_articles = []
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(extract_article_content, link): link for link in links}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    article_content = future.result()
                    all_articles.append(article_content)

                    # Ensure all datetime_obj fields are true datetime instances
                    for article in all_articles:
                        if "datetime_obj" in article:
                            if isinstance(article["datetime_obj"], str):
                                try:
                                    article["datetime_obj"] = datetime.fromisoformat(
                                        article["datetime_obj"]
                                    )
                                except ValueError:
                                    article["datetime_obj"] = datetime.min

                    # Sort newest first
                    sorted_articles = sorted(
                        all_articles,
                        key=lambda x: x.get("datetime_obj", datetime.min),
                        reverse=True,
                    )

                    # Serialise datetime objects before JSON dump
                    for article in sorted_articles:
                        if "datetime_obj" in article and isinstance(
                            article["datetime_obj"], datetime
                        ):
                            article["datetime_obj"] = article["datetime_obj"].isoformat()

                    output_data = {
                        "total_articles": len(sorted_articles),
                        "articles":       sorted_articles,
                    }
                    save_to_json(output_data, output_file)

                except Exception as e:
                    logging.error(f"Error processing link {url}: {e}")

    except Exception as e:
        logging.critical(f"Critical error during processing: {e}")

    return all_articles


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    input_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "output",
        "shrm_article_links_news20250119_024823.txt",
    )
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            links = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"File {input_file} not found. Ensure the file exists.")
        raise SystemExit(1)

    output_file = f"articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    logging.info("Starting article extraction process...")
    process_links(links, output_file)
    logging.info("Processing complete. All articles saved.")
