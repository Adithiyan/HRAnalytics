import os
import json
import datetime
import random
import time
import re
import datetime
from unidecode import unidecode
from playwright.sync_api import sync_playwright

# Configuration
INPUT_JSON = "scraped_links_views_uk.json"
OUTPUT_JSON = "cipd_article_data_views_uk.json"
LOGIN_URL = "https://www.cipd.org/login"
HOME_URL = "https://www.cipd.org/uk"
USERNAME = "Jade.yy423@gmail.com"
PASSWORD = "jadeCIPD123"
STORAGE_STATE_PATH = "storage_state.json"

def login(page):
    """
    Logs into the CIPD site.
    Assumes page is already at the login URL.
    """
    try:
        page.goto(LOGIN_URL)
        page.fill("input[name='username']", USERNAME)
        page.fill("input[name='password']", PASSWORD)

        # If there's an overlay from the cookie settings, remove it (rare).
        overlay_selector = "div.onetrust-pc-dark-filter"
        if page.locator(overlay_selector).is_visible():
            page.evaluate("document.querySelector(arguments[0]).style.display = 'none';", overlay_selector)

        page.click("button[type='submit']")
        page.wait_for_url(lambda url: url != LOGIN_URL, timeout=15000)
        print("Logged in successfully.")
    except Exception as e:
        print(f"Error during login: {e}")

def handle_cookie_banner(page):
    """
    Accepts the cookies if the banner is present, only once, at the start.
    """
    try:
        page.wait_for_selector("#onetrust-accept-btn-handler", timeout=10000)
        page.click("#onetrust-accept-btn-handler", timeout=5000)
        print("Accepted all cookies.")
    except Exception as e:
        # If the banner never appears or is already dismissed, just continue.
        print(f"Cookie banner not found or already accepted. ({e})")

def check_for_membership_lock(page):
    """
    Check if the page is locked for members only by detecting the relevant element.
    Returns True if the 'Members access only' content is visible.
    """
    locked_selector = ".member-messaging__title:has-text('Members access only')"
    return page.locator(locked_selector).count() > 0

def extract_article_content(page, url):
    """
    Visits the given URL, expands all accordions, and extracts article text/metadata.
    If membership-locked content is detected, logs in again, reloads page, and tries extraction.
    """
    article_data = {"url": url}


    def do_extraction():
        """
        Inner function to perform the actual extraction steps once the page is
        guaranteed to be accessible.
        """
        # 1) Expand all accordions so hidden content is visible
        accordion_buttons = page.locator("button.accordion__head")
        for i in range(accordion_buttons.count()):
            button = accordion_buttons.nth(i)
            if button.get_attribute("aria-expanded") == "false":
                button.click()
                page.wait_for_timeout(random.uniform(500, 1000))

        # 2) Title
        article_data["title"] = page.locator("h1").text_content() or ""
        article_data["title"] = article_data["title"].strip()

        # 3) Extract the date
        # Try multiple strategies to locate the date text:
        # Extract the date using multiple strategies
# Extract the date using multiple strategies
        # Extract the date using multiple strategies
        date_text = ""
        try:
            # First, define the selectors we want to check.
            # 1) Spans containing 'Published' or 'Last updated'.
            # 2) All spans in hero-member__info-item (to catch bare dates like "12 Aug, 2024").
            date_selectors = [
                "div.hero-member__info-item.group span.hero-member__info-item:has-text('Published')",
                "div.hero-member__info-item.group span.hero-member__info-item:has-text('Last updated')",
                "div.hero-member__info-item span.hero-member__info-item",
            ]

            # We'll use a regex that looks for patterns like "12 Aug, 2024" or "20 Dec, 2024"
            date_pattern = r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec),?\s+\d{4}\b"

            for selector in date_selectors:
                # Get all matching elements, not just the first
                elements = page.locator(selector)
                count = elements.count()
                if count == 0:
                    continue

                # Iterate over each matching element
                for i in range(count):
                    text_candidate = elements.nth(i).text_content().strip()
                    if not text_candidate:
                        continue

                    # Check if the text contains a date (via regex)
                    match = re.search(date_pattern, text_candidate, re.IGNORECASE)
                    if match:
                        date_text = match.group(0)
                        break

                # If we found date_text, no need to check further selectors
                if date_text:
                    break

            # If still no date found, try the fallback: examine the entire parent div
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

            # Clean up known prefixes like "Published:" or "Last updated:"
            if date_text:
                date_text = re.sub(r"^(Published|Last updated):\s*", "", date_text, flags=re.IGNORECASE)
                # Normalize whitespace and commas
                date_text = re.sub(r'\s+', ' ', date_text).strip()
                date_text = re.sub(r',\s*', ', ', date_text)

            # Store the final date string
            article_data["date"] = date_text
            article_data["year"] = None

            # Attempt to parse the date string into a datetime object
            if date_text:
                date_formats = [
                    "%d %b, %Y",  # e.g., "20 Dec, 2024"
                    "%d %b %Y",   # e.g., "20 Dec 2024"
                    "%B %d, %Y",  # e.g., "December 20, 2024"
                    "%d-%b-%Y",   # e.g., "20-Dec-2024"
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



        # 4) Category
        try:
            category = page.locator("span.hero-member__info-item.label")
            if category.count() > 0:
                article_data["category"] = category.text_content().strip()
            else:
                article_data["category"] = ""
        except Exception:
            article_data["category"] = ""

        # 5) Tags
        try:
            tags = page.locator("div.hero-member__info-tags .tag").all_text_contents()
            article_data["tags"] = [tag.strip() for tag in tags]
        except Exception:
            article_data["tags"] = []

        # 6) Description
        try:
            description = page.locator("div.hero-section__description p") \
                            .text_content().strip()
            article_data["description"] = description
        except Exception:
            article_data["description"] = ""

        # 7) SUMMARY (from <section class="page-intro">…</section>)
        try:
            summary_locator = page.locator("section.page-intro")
            # Gather *all* text inside `.page-intro` in case there are multiple <p> or <em> tags
            summary_texts = summary_locator.all_text_contents()
            summary_combined = " ".join(t.strip() for t in summary_texts if t.strip())
            article_data["summary"] = summary_combined.strip()
        except Exception:
            article_data["summary"] = ""

        # 8) FULL TEXT: everything *after* .page-intro, ignoring sections with .share or .curated-cards__wrapper
        #    - Also include accordion section titles (accordion__title / accordion_title).
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

            final_text = "\n".join(full_text_lines).strip()
            final_text = unidecode(final_text)

            # 8a) Exclude unwanted cookie-consent block that may appear at the end:
            for marker in ["Manage Consent Preferences", "Strictly Necessary Cookies"]:
                pos = final_text.lower().find(marker.lower())
                if pos != -1:
                    final_text = final_text[:pos].strip()
                    break  # Stop at the first marker we find

            article_data["full_text"] = final_text

        except Exception:
            article_data["full_text"] = ""

        # 9) LINKS (ignoring .share or .curated-cards__wrapper sections)
        try:
            link_elements = page.locator("""
            section:not(.page-intro):not(:has(.share)):not(:has(.curated-cards__wrapper)) a
            """).element_handles()
            links = []
            for element in link_elements:
                href = element.get_attribute("href")
                if href:
                    links.append(href)
            article_data["links"] = links
        except Exception:
            article_data["links"] = []

        # 10) Directory (breadcrumb)
        try:
            breadcrumb_items = page.locator("ol.breadcrumb li span[itemprop='name']") \
                                .all_text_contents()
            article_data["directory"] = " > ".join(item.strip() for item in breadcrumb_items)
        except Exception:
            article_data["directory"] = ""



    try:
        # Load the article page
        page.goto(url, timeout=20000)
        page.wait_for_load_state("networkidle")

        # If session expired, login again
        if page.locator("input[name='username']").count() > 0:
            print("Session expired or not logged in. Logging in again.")
            login(page)
            page.goto(url, timeout=20000)
            page.wait_for_load_state("networkidle")

        # Check for membership lock
        if check_for_membership_lock(page):
            print("Content is for members only. Re-logging in to unlock.")
            login(page)
            page.goto(url, timeout=20000)
            page.wait_for_load_state("networkidle")

        time.sleep(random.uniform(1, 3))
        do_extraction()

    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        article_data["full_text"] = ""
        article_data["links"] = []

    return article_data

def parse_date(article):
    """
    Safely parse the date string in the article data for sorting purposes.
    """
    try:
        return datetime.datetime.strptime(article.get("date", "01 Jan, 1900"), "%d %b, %Y")
    except ValueError:
        return datetime.datetime(1900, 1, 1)

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        # Load session if storage state exists; otherwise create a new one.
        if os.path.exists(STORAGE_STATE_PATH):
            context = browser.new_context(storage_state=STORAGE_STATE_PATH)
            print("Loaded existing storage state.")
        else:
            context = browser.new_context()
            print("No existing storage state. Creating a new session.")

        page = context.new_page()

        try:
            # 1) Go to CIPD login page and log in first.
            #page.goto(LOGIN_URL)
            #page.wait_for_load_state("load")
            #login(page)

            # 2) Next, go to the homepage to handle cookie banner once.
            page.goto(HOME_URL)
            page.wait_for_load_state("networkidle")
            handle_cookie_banner(page)

            # Save storage state after successful login + cookie acceptance
            context.storage_state(path=STORAGE_STATE_PATH)

            # 3) Read input JSON containing article URLs
            with open(INPUT_JSON, "r") as f:
                scraped_data = json.load(f)

            articles = []
            # 4) For each URL, extract the content using the same session
            for main_link, data in scraped_data.items():
                for url in data["articles"]:
                    print(f"Extracting content from: {url}")
                    article_data = extract_article_content(page, url)
                    articles.append(article_data)

            # 5) Sort articles by date (descending)
            articles.sort(key=lambda a: parse_date(a), reverse=True)

            # 6) Write output JSON
            output_data = {
                "total_articles": len(articles),
                "articles": articles
            }
            with open(OUTPUT_JSON, "w") as out_f:
                json.dump(output_data, out_f, indent=4)

            print(f"Extraction completed. Data saved to {OUTPUT_JSON}")

        finally:
            # Save the final storage state and close the browser
            context.storage_state(path=STORAGE_STATE_PATH)
            browser.close()

if __name__ == "__main__":
    main()
