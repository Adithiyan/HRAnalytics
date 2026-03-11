"""
shrm/link_collector.py — Stage 1: Collect article URLs from SHRM.

Navigates the paginated SHRM Topics & Tools search hub using Playwright,
extracts article links from each page's Coveo atomic-result shadow DOM
components, and writes all unique links to a timestamped text file.

Input:  None (starts from SHRM_TOPICS_URL defined in config)
Output: output/shrm_article_links_news<timestamp>.txt

Usage:
    python -m shrm.link_collector          # from src/
    python src/shrm/link_collector.py      # from project root
"""

import os
import sys
import time
import random
from datetime import datetime
from playwright.sync_api import sync_playwright

# Allow running as a standalone script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, SHRM_TOPICS_URL


def handle_cookies(page):
    """
    Dismiss the OneTrust cookie consent banner by clicking 'Reject All'.

    Waits up to 5 s for the banner; silently continues if it is absent.
    After handling the banner a 30-second pause lets the operator verify
    the page state and active filters before automated pagination begins.
    """
    print("Checking for cookie consent banner...")
    try:
        page.wait_for_selector('#onetrust-reject-all-handler', timeout=5000)
        reject_button = page.query_selector('#onetrust-reject-all-handler')
        if reject_button:
            reject_button.click()
            print("Cookie consent rejected.")
            page.wait_for_timeout(1000)
        else:
            print("Cookie consent banner not found.")
    except Exception as e:
        print(f"No cookie consent banner detected: {e}")
    finally:
        print("Waiting for 30 seconds to let user verify filters...")
        time.sleep(30)


def collect_article_links(url, max_pages=1380, retries=3):
    """
    Collect unique article URLs from a paginated SHRM search results page.

    The SHRM search hub renders results inside <atomic-result> web components
    with shadow DOM.  Each component's shadow root contains an <a> whose href
    is the canonical article URL.  Pagination is driven by the <atomic-pager>
    shadow-root 'Next' button.

    Args:
        url (str): Starting URL — typically SHRM_TOPICS_URL.
        max_pages (int): Maximum number of result pages to visit (default 1380,
                         which covers the full SHRM knowledge base as of 2025).
        retries (int): Per-page retry attempts before giving up.

    Returns:
        list[str]: Ordered list of unique article URLs collected.

    Side-effects:
        Writes links to output/shrm_article_links_news<timestamp>.txt.
    """
    all_links  = []
    seen_links = set()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page    = browser.new_page()
        print(f"Navigating to {url}")
        page.goto(url)

        handle_cookies(page)

        for current_page in range(1, max_pages + 1):
            print(f"\n--- Processing Page {current_page}/{max_pages} ---")
            success = False

            for attempt in range(retries):
                try:
                    page.wait_for_selector('atomic-result', timeout=20000)
                    results = page.query_selector_all('atomic-result')

                    page_links = []
                    for result in results:
                        try:
                            shadow_root = result.evaluate_handle('el => el.shadowRoot')
                            link = shadow_root.eval_on_selector('a', 'el => el.href')
                            if link and link.strip() and link.strip() not in seen_links:
                                seen_links.add(link.strip())
                                page_links.append(link.strip())
                                all_links.append(link.strip())
                        except Exception as e:
                            print(f"Error extracting link: {e}")

                    print(f"Total unique links found on page {current_page}: {len(page_links)}")

                    # Advance to next page via the atomic-pager shadow root
                    pager = page.query_selector('atomic-pager')
                    if pager:
                        shadow_root = pager.evaluate_handle('el => el.shadowRoot')
                        next_button = shadow_root.eval_on_selector(
                            'button[aria-label="Next"]', 'el => el'
                        )
                        if next_button and shadow_root.eval_on_selector(
                            'button[aria-label="Next"]', 'el => !el.disabled'
                        ):
                            shadow_root.eval_on_selector(
                                'button[aria-label="Next"]', 'el => el.click()'
                            )
                            time.sleep(random.uniform(1.5, 3))
                            success = True
                            break
                        else:
                            print("No more pages to navigate.")
                            success = True
                            break
                    else:
                        print("Pagination controls not found.")
                        success = True
                        break

                except Exception as e:
                    print(f"Error on page {current_page}, attempt {attempt + 1}: {e}")
                    time.sleep(2)

            if not success:
                print(f"Failed to process page {current_page} after {retries} retries.")
                break

        browser.close()

    # Persist collected links
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"shrm_article_links_news{timestamp}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_links))
    print(f"Links saved to {output_file}")

    return all_links


if __name__ == "__main__":
    collected_links = collect_article_links(SHRM_TOPICS_URL)
