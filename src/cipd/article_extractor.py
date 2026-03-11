"""
cipd/article_extractor.py — Stage 3: Extract full article content from CIPD URLs.

Uses Playwright (Chromium) with a persisted storage-state session so that the
authenticated context survives across articles without repeated logins.

For each URL the script:
  1. Navigates to the article page and re-logs in if the session has expired.
  2. Expands all accordion sections so hidden content is included.
  3. Scrapes title, date, category, tags, description, summary, full text,
     internal links, and breadcrumb directory.

Date parsing supports multiple CIPD formats (e.g. "12 Aug, 2024").
The output JSON is sorted newest-first and includes a total_articles count.

Input:  cipd/scraped_links_views_uk.json  (written by cipd/article_links.py)
        storage_state.json               (Playwright session — created if absent)
Output: cipd/cipd_article_data_views_uk.json

Usage:
    python -m cipd.article_extractor      # from src/
    python src/cipd/article_extractor.py  # from project root
"""

import os
import sys
import json
import datetime
import random
import time
import re

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover - dependency may be absent in lean envs
    def unidecode(text):
        return text
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CIPD_USERNAME, CIPD_PASSWORD,
    CIPD_LOGIN_URL, CIPD_HOME_URL,
    CIPD_STORAGE_STATE, CIPD_ARTICLE_LINKS_FILE, CIPD_ARTICLE_DATA_FILE,
)

# Expose HOME_URL so run_2025.py can import it without repeating the constant
HOME_URL = CIPD_HOME_URL


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def login(page):
    """
    Fill and submit the CIPD login form on the current page.

    Waits until the URL changes away from the login page (up to 15 s) to
    confirm successful authentication.  Removes the OneTrust overlay if it
    obstructs the submit button.
    """
    try:
        page.goto(CIPD_LOGIN_URL)
        page.fill("input[name='username']", CIPD_USERNAME)
        page.fill("input[name='password']", CIPD_PASSWORD)

        overlay_selector = "div.onetrust-pc-dark-filter"
        if page.locator(overlay_selector).is_visible():
            page.evaluate(
                "document.querySelector(arguments[0]).style.display = 'none';",
                overlay_selector,
            )

        page.click("button[type='submit']")
        page.wait_for_url(lambda url: url != CIPD_LOGIN_URL, timeout=15000)
        print("Logged in successfully.")
    except Exception as e:
        print(f"Error during login: {e}")


def handle_cookie_banner(page):
    """
    Accept the CIPD cookie banner if it appears (waits up to 10 s).

    Only called once at session start; silently continues if absent.
    """
    try:
        page.wait_for_selector("#onetrust-accept-btn-handler", timeout=10000)
        page.click("#onetrust-accept-btn-handler", timeout=5000)
        print("Accepted all cookies.")
    except Exception as e:
        print(f"Cookie banner not found or already accepted. ({e})")


def check_for_membership_lock(page):
    """
    Return True if the page shows a 'Members access only' gate.

    Used to detect when a re-login is needed to access member content.
    """
    locked_selector = ".member-messaging__title:has-text('Members access only')"
    return page.locator(locked_selector).count() > 0


# ---------------------------------------------------------------------------
# Article extraction
# ---------------------------------------------------------------------------

def extract_article_content(page, url):
    """
    Navigate to `url` and extract structured content from a CIPD article.

    Automatically re-authenticates if the session has expired or the content
    is behind a membership gate.

    Extracted fields:
        url, title, date, year, category, tags, description, summary,
        full_text (accordions expanded), links, directory (breadcrumb)

    Args:
        page: Active Playwright Page object with an authenticated context.
        url (str): Absolute CIPD article URL.

    Returns:
        dict: Structured article data.
    """
    article_data = {"url": url}

    def do_extraction():
        """Perform the actual DOM scraping once the page is confirmed accessible."""

        # Expand all collapsed accordions
        accordion_buttons = page.locator("button.accordion__head")
        for i in range(accordion_buttons.count()):
            button = accordion_buttons.nth(i)
            if button.get_attribute("aria-expanded") == "false":
                button.click()
                page.wait_for_timeout(random.uniform(500, 1000))

        # Title
        article_data["title"] = (page.locator("h1").text_content() or "").strip()

        # Date — tries multiple selectors and regex patterns
        date_text = ""
        try:
            date_selectors = [
                "div.hero-member__info-item.group span.hero-member__info-item:has-text('Published')",
                "div.hero-member__info-item.group span.hero-member__info-item:has-text('Last updated')",
                "div.hero-member__info-item span.hero-member__info-item",
            ]
            date_pattern = r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec),?\s+\d{4}\b"

            for selector in date_selectors:
                elements = page.locator(selector)
                if elements.count() == 0:
                    continue
                for i in range(elements.count()):
                    text_candidate = elements.nth(i).text_content().strip()
                    if not text_candidate:
                        continue
                    match = re.search(date_pattern, text_candidate, re.IGNORECASE)
                    if match:
                        date_text = match.group(0)
                        break
                if date_text:
                    break

            # Fallback: scan the entire parent div
            if not date_text:
                try:
                    parent_div = page.locator("div.hero-member__info-item.group").first
                    if parent_div.count() > 0:
                        full_text = parent_div.text_content().strip()
                        match = re.search(date_pattern, full_text, re.IGNORECASE)
                        if match:
                            date_text = match.group(0)
                except Exception:
                    pass

            if date_text:
                date_text = re.sub(r"^(Published|Last updated):\s*", "", date_text, flags=re.IGNORECASE)
                date_text = re.sub(r'\s+', ' ', date_text).strip()
                date_text = re.sub(r',\s*', ', ', date_text)

            article_data["date"] = date_text
            article_data["year"] = None

            if date_text:
                date_formats = [
                    "%d %b, %Y",  # "20 Dec, 2024"
                    "%d %b %Y",   # "20 Dec 2024"
                    "%B %d, %Y",  # "December 20, 2024"
                    "%d-%b-%Y",   # "20-Dec-2024"
                ]
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.datetime.strptime(date_text, fmt)
                        article_data["year"] = parsed_date.year
                        break
                    except ValueError:
                        continue

        except Exception as e:
            print(f"Error extracting date: {str(e)}")

        # Category label
        try:
            category = page.locator("span.hero-member__info-item.label")
            article_data["category"] = (
                category.text_content().strip() if category.count() > 0 else ""
            )
        except Exception:
            article_data["category"] = ""

        # Tags
        try:
            tags = page.locator("div.hero-member__info-tags .tag").all_text_contents()
            article_data["tags"] = [tag.strip() for tag in tags]
        except Exception:
            article_data["tags"] = []

        # Hero description
        try:
            article_data["description"] = (
                page.locator("div.hero-section__description p").text_content().strip()
            )
        except Exception:
            article_data["description"] = ""

        # Summary (page-intro section)
        try:
            summary_texts  = page.locator("section.page-intro").all_text_contents()
            article_data["summary"] = " ".join(t.strip() for t in summary_texts if t.strip()).strip()
        except Exception:
            article_data["summary"] = ""

        # Full text (all sections except page-intro, share, curated-cards)
        try:
            full_text_elements = page.locator("""
            section:not(.page-intro):not(:has(.share)):not(:has(.curated-cards__wrapper))
            h1,
            section:not(.page-intro):not(:has(.share)):not(:has(.curated-cards__wrapper))
            h2,
            section:not(.page-intro):not(:has(.share)):not(:has(.curated-cards__wrapper))
            h3,
            section:not(.page-intro):not(:has(.share)):not(:has(.curated-cards__wrapper))
            h4,
            section:not(.page-intro):not(:has(.share)):not(:has(.curated-cards__wrapper))
            h5,
            section:not(.page-intro):not(:has(.share)):not(:has(.curated-cards__wrapper))
            h6,
            section:not(.page-intro):not(:has(.share)):not(:has(.curated-cards__wrapper))
            p,
            section:not(.page-intro):not(:has(.share)):not(:has(.curated-cards__wrapper))
            li,
            section:not(.page-intro):not(:has(.share)):not(:has(.curated-cards__wrapper))
            span.accordion__title,
            section:not(.page-intro):not(:has(.share)):not(:has(.curated-cards__wrapper))
            span.accordion_title
            """)

            full_text_lines = []
            for elem_handle in full_text_elements.element_handles():
                text = elem_handle.inner_text().strip()
                if text:
                    full_text_lines.append(text)

            final_text = unidecode("\n".join(full_text_lines).strip())

            # Strip trailing cookie-consent boilerplate if captured
            for marker in ["Manage Consent Preferences", "Strictly Necessary Cookies"]:
                pos = final_text.lower().find(marker.lower())
                if pos != -1:
                    final_text = final_text[:pos].strip()
                    break

            article_data["full_text"] = final_text

        except Exception:
            article_data["full_text"] = ""

        # Internal links
        try:
            link_elements = page.locator(
                "section:not(.page-intro):not(:has(.share)):not(:has(.curated-cards__wrapper)) a"
            ).element_handles()
            article_data["links"] = [
                el.get_attribute("href")
                for el in link_elements
                if el.get_attribute("href")
            ]
        except Exception:
            article_data["links"] = []

        # Breadcrumb directory
        try:
            breadcrumb_items = (
                page.locator("ol.breadcrumb li span[itemprop='name']").all_text_contents()
            )
            article_data["directory"] = " > ".join(item.strip() for item in breadcrumb_items)
        except Exception:
            article_data["directory"] = ""

    # --- Navigate and extract, re-authenticating if needed ---
    try:
        page.goto(url, timeout=20000)
        page.wait_for_load_state("networkidle")

        if page.locator("input[name='username']").count() > 0:
            print("Session expired. Logging in again.")
            login(page)
            page.goto(url, timeout=20000)
            page.wait_for_load_state("networkidle")

        if check_for_membership_lock(page):
            print("Content gated. Re-logging in.")
            login(page)
            page.goto(url, timeout=20000)
            page.wait_for_load_state("networkidle")

        time.sleep(random.uniform(1, 3))
        do_extraction()

    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        article_data["full_text"] = ""
        article_data["links"]     = []

    return article_data


# ---------------------------------------------------------------------------
# Date sorting helper
# ---------------------------------------------------------------------------

def parse_date(article):
    """
    Return a datetime for sorting.  Falls back to 1 Jan 1900 on parse error.
    """
    try:
        return datetime.datetime.strptime(article.get("date", "01 Jan, 1900"), "%d %b, %Y")
    except ValueError:
        return datetime.datetime(1900, 1, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the full CIPD article extraction pipeline."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        context = (
            browser.new_context(storage_state=CIPD_STORAGE_STATE)
            if os.path.exists(CIPD_STORAGE_STATE)
            else browser.new_context()
        )
        print("Loaded existing storage state." if os.path.exists(CIPD_STORAGE_STATE)
              else "No existing storage state. Creating a new session.")

        page = context.new_page()

        try:
            page.goto(HOME_URL)
            page.wait_for_load_state("networkidle")
            handle_cookie_banner(page)
            context.storage_state(path=CIPD_STORAGE_STATE)

            with open(CIPD_ARTICLE_LINKS_FILE, "r") as f:
                scraped_data = json.load(f)

            articles = []
            for main_link, data in scraped_data.items():
                for url in data["articles"]:
                    print(f"Extracting content from: {url}")
                    article_data = extract_article_content(page, url)
                    articles.append(article_data)

            articles.sort(key=lambda a: parse_date(a), reverse=True)

            output_data = {"total_articles": len(articles), "articles": articles}
            with open(CIPD_ARTICLE_DATA_FILE, "w") as out_f:
                json.dump(output_data, out_f, indent=4)

            print(f"Extraction completed. Data saved to {CIPD_ARTICLE_DATA_FILE}")

        finally:
            context.storage_state(path=CIPD_STORAGE_STATE)
            browser.close()


if __name__ == "__main__":
    main()
