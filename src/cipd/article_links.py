"""
cipd/article_links.py — Stage 2: Collect article URLs from each CIPD category.

Reads the top-level category URLs produced by Stage 1 (CIPD_MAIN_LINKS_FILE),
visits each one using Playwright (same session as Stage 3), and iterates through
all pagination pages to harvest every 'div.card.card--full' article link.

Uses Playwright instead of Selenium for reliable DOM interaction and consistent
session management via storage_state.json.

Input:  cipd/main_links_views_uk.txt   (written by cipd/main_links.py)
        storage_state.json             (Playwright session)
Output: cipd/scraped_links_views_uk.json
        storage_state.json             (refreshed)

Usage:
    python -m cipd.article_links      # from src/
    python src/cipd/article_links.py  # from project root
"""

import os
import sys
import json
import time
import random
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CIPD_USERNAME, CIPD_PASSWORD,
    CIPD_LOGIN_URL, CIPD_HOME_URL,
    CIPD_STORAGE_STATE, CIPD_MAIN_LINKS_FILE, CIPD_ARTICLE_LINKS_FILE,
)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def login(page):
    """
    Fill and submit the CIPD login form.

    Waits until the URL changes away from the login page to confirm success.
    Removes the OneTrust overlay if it obstructs the submit button.
    """
    try:
        page.goto(CIPD_LOGIN_URL)
        page.fill("input[name='username']", CIPD_USERNAME)
        page.fill("input[name='password']", CIPD_PASSWORD)

        overlay = "div.onetrust-pc-dark-filter"
        if page.locator(overlay).is_visible():
            page.evaluate("document.querySelector(arguments[0]).style.display='none'", overlay)

        page.click("button[type='submit']")
        page.wait_for_url(lambda url: url != CIPD_LOGIN_URL, timeout=15000)
        print("Logged in successfully.")
    except Exception as e:
        print(f"Login error: {e}")


def handle_cookie_banner(page):
    """Accept the CIPD cookie banner if present (waits up to 10 s)."""
    try:
        page.wait_for_selector("#onetrust-accept-btn-handler", timeout=10000)
        page.click("#onetrust-accept-btn-handler", timeout=5000)
        print("Accepted cookies.")
    except Exception:
        pass  # Banner absent or already dismissed


# ---------------------------------------------------------------------------
# Link extraction
# ---------------------------------------------------------------------------

def extract_article_links_from_page(page):
    """
    Return all 'div.card.card--full' article hrefs visible on the current page.

    Args:
        page: Active Playwright Page.

    Returns:
        list[str]: Article URLs found on this page.
    """
    links = []
    try:
        cards = page.locator("div.card.card--full")
        count = cards.count()
        for i in range(count):
            try:
                link_el = cards.nth(i).locator("a.link--arrow")
                if link_el.count() > 0:
                    # Use .evaluate to get the fully-resolved absolute URL
                    href = link_el.first.evaluate("el => el.href")
                    if href:
                        links.append(href)
            except Exception as e:
                print(f"  Error extracting card link: {e}")
    except Exception as e:
        print(f"  Error querying cards: {e}")
    return links


def scrape_category(page, category_url):
    """
    Collect all article links from a CIPD category page, following pagination.

    Args:
        page: Authenticated Playwright Page.
        category_url (str): CIPD category URL to scrape.

    Returns:
        list[str]: All unique article URLs found across all pages of this category.
    """
    all_links = []
    seen     = set()

    print(f"\nScraping: {category_url}")
    try:
        page.goto(category_url, timeout=20000)
        page.wait_for_load_state("networkidle")
    except Exception as e:
        print(f"  Failed to load {category_url}: {e}")
        return all_links

    # Re-login if session expired
    if page.locator("input[name='username']").count() > 0:
        print("  Session expired — logging in.")
        login(page)
        page.goto(category_url, timeout=20000)
        page.wait_for_load_state("networkidle")

    page_num = 0
    while True:
        page_num += 1
        time.sleep(random.uniform(1, 2))

        links = extract_article_links_from_page(page)
        new_links = [l for l in links if l not in seen]
        seen.update(new_links)
        all_links.extend(new_links)
        print(f"  Page {page_num}: {len(new_links)} new links (total {len(all_links)})")

        # Try to advance to the next page
        try:
            next_btn = page.locator("button.pagination__dir--next")
            if next_btn.count() > 0 and next_btn.first.is_enabled():
                next_btn.first.evaluate("el => el.scrollIntoView(true)")
                time.sleep(0.5)
                next_btn.first.click()
                page.wait_for_load_state("networkidle")
                time.sleep(random.uniform(1, 2))
            else:
                print(f"  No more pages for {category_url}")
                break
        except Exception as e:
            print(f"  Pagination ended: {e}")
            break

    return all_links


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Run CIPD Stage 2: collect article links from all category URLs.

    Reads category URLs from CIPD_MAIN_LINKS_FILE, scrapes each with Playwright,
    and writes the consolidated result to CIPD_ARTICLE_LINKS_FILE.
    """
    with open(CIPD_MAIN_LINKS_FILE, "r") as f:
        main_links = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(main_links)} category URLs from {CIPD_MAIN_LINKS_FILE}")

    scraped_data = {}

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

            for category_url in main_links:
                article_links = scrape_category(page, category_url)
                scraped_data[category_url] = {
                    "articles":       article_links,
                    "total_articles": len(article_links),
                }

            with open(CIPD_ARTICLE_LINKS_FILE, "w") as f:
                json.dump(scraped_data, f, indent=4)
            print(f"\nDone. Saved to {CIPD_ARTICLE_LINKS_FILE}")

        finally:
            context.storage_state(path=CIPD_STORAGE_STATE)
            browser.close()


if __name__ == "__main__":
    main()
