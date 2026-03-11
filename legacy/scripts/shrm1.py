import os
import time
import random
from datetime import datetime
from playwright.sync_api import sync_playwright


def handle_cookies(page):
    """
    Handle the cookie consent banner by clicking the "Reject All" button if present.
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
        print("Waiting for 15 seconds to let user verify filters...")
        time.sleep(30)

#max pages = 1380
def collect_article_links(url, max_pages=1380, retries=3):
    """
    Collect article links from paginated pages without duplicates.
    
    Args:
        url (str): Starting URL for collection.
        max_pages (int): Number of pages to process.
        retries (int): Number of retries for each page in case of failures.

    Returns:
        list: All links collected in order.
    """
    all_links = []
    seen_links = set()  # Set to track seen links
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        print(f"Navigating to {url}")
        page.goto(url)

        # Handle cookies
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

                    # Handle pagination
                    pager = page.query_selector('atomic-pager')
                    if pager:
                        shadow_root = pager.evaluate_handle('el => el.shadowRoot')
                        next_button = shadow_root.eval_on_selector('button[aria-label="Next"]', 'el => el')

                        if next_button and shadow_root.eval_on_selector('button[aria-label="Next"]', 'el => !el.disabled'):
                            shadow_root.eval_on_selector('button[aria-label="Next"]', 'el => el.click()')
                            time.sleep(random.uniform(1.5, 3))  # Random delay
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

    # Save links to a text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"shrm_article_links_news{timestamp}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_links))
    print(f"Links saved to {output_file}")

    return all_links



if __name__ == "__main__":
    target_url = "https://www.shrm.org/topics-tools"
    collected_links = collect_article_links(target_url)
