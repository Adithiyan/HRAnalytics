"""
cipd/blog_scraper.py — Static scraper for CIPD community blog articles.

Uses requests + BeautifulSoup (no browser automation) to fetch CIPD community
blog pages and extract title, author, full text, links, summary, and
breadcrumb directory.  Multiple CSS selector strategies are attempted for
each field to handle layout variations across blog post templates.

Input:  scraped_links_cipd_community_target.json
            Expected structure: { "articles": ["https://...", ...] }
Output: cipd_articles_static_scrape.json

Usage:
    python -m cipd.blog_scraper      # from src/
    python src/cipd/blog_scraper.py  # from project root
"""

import os
import sys
import json
import time
import random
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CIPD_BLOG_INPUT_FILE, CIPD_BLOG_OUTPUT_FILE

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Debug utility
# ---------------------------------------------------------------------------

def debug_page_structure(soup, url):
    """
    Print a diagnostic summary of common content-container selectors found on
    the page.  Useful when a new blog template is encountered.

    Args:
        soup (BeautifulSoup): Parsed page.
        url (str): Source URL (for display only).
    """
    print(f"Debugging page structure for: {url}")

    for selector in [
        "div.content.full > div.content", "div.content.full", "div.content",
        ".content", "article", ".post-content", ".blog-post", "main", "#main-content",
    ]:
        elements = soup.select(selector)
        print(f"  '{selector}': {len(elements)} elements found")
        if elements:
            print(f"    Preview: {elements[0].get_text(strip=True)[:100]}...")

    print("Title candidates:")
    for selector in ["div.content.full > h3.name", "h1", "h2", "h3", ".title", ".post-title"]:
        for i, elem in enumerate(soup.select(selector)[:3]):
            print(f"  '{selector}' [{i}]: {elem.get_text(strip=True)[:50]}...")


# ---------------------------------------------------------------------------
# Single-article extraction
# ---------------------------------------------------------------------------

def extract_article_content(url):
    """
    Fetch one CIPD community blog URL and extract its structured content.

    Selectors are tried in priority order for each field; the first match wins.
    The author element is removed from the DOM before paragraph extraction to
    prevent it from appearing inside full_text.

    Fields returned:
        url, title, author, full_text, links, summary, tags, category,
        directory  (and optionally 'error' on failure)

    Args:
        url (str): Absolute URL of the community blog post.

    Returns:
        dict: Extracted article data.
    """
    print(f"\nFetching: {url}")
    article = {"url": url}

    try:
        time.sleep(random.uniform(1, 2))
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Title — try selectors in priority order
        title = ""
        for selector in [
            "div.content.full > h3.name", "h1", "h2.post-title",
            ".post-title", "h3.name", "title",
        ]:
            title_el = soup.select_one(selector)
            if title_el and title_el.get_text(strip=True):
                title = title_el.get_text(strip=True)
                print(f"Title found with '{selector}': {title[:50]}...")
                break
        article["title"] = title

        # Main content container
        main_content = None
        for selector in [
            "div.content.full > div.content", "div.content.full", "div.content",
            ".content", "article", ".post-content", ".blog-content", "main",
        ]:
            candidate = soup.select_one(selector)
            if candidate:
                main_content = candidate
                print(f"Content found with '{selector}'")
                break

        if not main_content:
            print("No main content container found — falling back to <body>")
            main_content = soup.find("body")
            if not main_content:
                raise Exception("No content container found")

        # Author — several patterns tried; element removed before text extraction
        author = ""
        if main_content:
            author_patterns = [
                lambda c: c.find("p", string=lambda t: t and "By " in t),
                lambda c: c.find("p", lambda tag: tag.find("strong") and "By " in tag.get_text()),
                lambda c: c.find(class_="author"),
                lambda c: c.find(class_="by-author"),
                lambda c: c.select_one(".author-name"),
            ]
            for pattern in author_patterns:
                try:
                    author_elem = pattern(main_content)
                    if author_elem:
                        author_text = author_elem.get_text(strip=True)
                        author = author_text.replace("By ", "").strip() if "By " in author_text \
                                 else author_text.strip()
                        author_elem.decompose()
                        break
                except Exception:
                    continue
        article["author"] = author

        # Full text — paragraphs longer than 10 chars joined with double newline
        full_text = ""
        if main_content:
            paragraphs = main_content.find_all("p")
            if paragraphs:
                paragraph_texts = [
                    p.get_text(strip=True) for p in paragraphs
                    if len(p.get_text(strip=True)) > 10
                ]
                full_text = "\n\n".join(paragraph_texts)
            else:
                full_text = main_content.get_text(separator="\n", strip=True)

            if full_text:
                full_text = unidecode(full_text)
                full_text = "\n".join(
                    line.strip() for line in full_text.split("\n") if line.strip()
                )
        article["full_text"] = full_text

        # Links (relative → absolute; anchor-only links excluded)
        links = []
        if main_content:
            for a in main_content.find_all("a", href=True):
                href = a.get("href", "").strip()
                if href and not href.startswith("#"):
                    if href.startswith("/"):
                        href = "https://community.cipd.co.uk" + href
                    links.append(href)
        article["links"] = list(set(links))

        # Summary — first 250 chars of full text
        summary = full_text[:250].strip() if full_text else ""
        if len(full_text) > 250:
            summary += "..."
        article["summary"] = summary

        article["tags"]     = []
        article["category"] = ""

        # Breadcrumb directory
        directory = ""
        directory_container = soup.select_one("div.row ul")
        if directory_container:
            breadcrumb_items = []
            for li in directory_container.find_all("li"):
                link = li.find("a")
                text = link.get_text(strip=True) if link else li.get_text(strip=True)
                if text:
                    breadcrumb_items.append(text)
            breadcrumb_items = [item for item in breadcrumb_items if item.lower() != "home"]
            directory = " > ".join(breadcrumb_items)
            if directory:
                print(f"Directory found: {directory}")
        article["directory"] = directory

        print(f"Extracted: '{title[:40]}...', {len(full_text)} chars, {len(links)} links")
        if full_text:
            print(f"Content preview: {full_text[:150].replace(chr(10), ' ')}...")

    except Exception as e:
        print(f"Failed to fetch {url}: {str(e)}")
        article.update({
            "title": "", "author": "", "full_text": "", "summary": "",
            "description": "", "links": [], "tags": [], "category": "",
            "directory": "", "error": str(e),
        })

    return article


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Read URL list, extract all articles, and write consolidated JSON output."""
    print("Starting CIPD blog scraper...")

    try:
        with open(CIPD_BLOG_INPUT_FILE, "r") as f:
            url_data = json.load(f)
    except FileNotFoundError:
        print(f"Input file '{CIPD_BLOG_INPUT_FILE}' not found!")
        return
    except json.JSONDecodeError:
        print(f"Invalid JSON in '{CIPD_BLOG_INPUT_FILE}'!")
        return

    urls = url_data.get("articles", [])
    if not urls:
        print("No URLs found in the input file!")
        return

    print(f"Processing {len(urls)} URLs...")
    results = []

    for i, url in enumerate(urls, 1):
        print(f"\n--- [{i}/{len(urls)}] ---")
        results.append(extract_article_content(url))

        if i % 5 == 0:
            successful = sum(1 for r in results if r.get("full_text"))
            print(f"Progress: {i}/{len(urls)} processed, {successful} successful extractions")

    try:
        with open(CIPD_BLOG_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_articles":         len(results),
                    "successful_extractions": sum(1 for r in results if r.get("full_text")),
                    "extraction_timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
                    "articles":               results,
                },
                f, indent=2, ensure_ascii=False,
            )
        successful = sum(1 for r in results if r.get("full_text"))
        print(f"\nDone! Results saved to '{CIPD_BLOG_OUTPUT_FILE}'")
        print(f"Summary: {successful}/{len(results)} articles successfully extracted")
    except Exception as e:
        print(f"Failed to save results: {e}")


if __name__ == "__main__":
    main()
