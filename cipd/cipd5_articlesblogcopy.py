import os
import json
import datetime
import random
import time
import asyncio
from urllib.parse import urlparse
import hashlib

from unidecode import unidecode
from playwright.async_api import async_playwright

# Configuration
INPUT_JSON = "scraped_links_cipd_community.json"
OUTPUT_JSON = "cipd_articles__cipd_community_new.json"
LOGIN_URL = "https://auth.cipd.org/u/login"
HOME_URL = "https://www.peoplemanagement.co.uk"
USERNAME = "Jade.yy423@gmail.com"
PASSWORD = "jadeCIPD123"

STORAGE_STATE_PATH = "storage_state.json"
CACHE_DIR = "article_cache"
MAX_CONCURRENT_PAGES = 3  # Control concurrency to avoid being blocked
REQUEST_TIMEOUT = 15000  # 15 seconds
PAGE_NAVIGATION_TIMEOUT = 20000  # 20 seconds
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(url):
    """Generate a unique cache filename based on URL"""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{url_hash}.json")

async def load_from_cache(url):
    """Try to load article data from cache"""
    cache_path = get_cache_path(url)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            os.remove(cache_path)  # Remove corrupted cache file
    return None

def save_to_cache(url, article_data):
    """Save article data to cache"""
    cache_path = get_cache_path(url)
    with open(cache_path, 'w') as f:
        json.dump(article_data, f)

async def login(page):
    """Login to CIPD website with retry logic"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Go to the homepage
            await page.goto(HOME_URL, timeout=PAGE_NAVIGATION_TIMEOUT)
            await page.wait_for_load_state("networkidle")
            await handle_cookie_banner(page)

            # Check if already logged in
            account_element = page.locator("text=My Account")
            if await account_element.count() > 0 and await account_element.is_visible():
                print("Already logged in.")
                return True

            # Click the CIPD login link
            await page.click("#myAccount a[href*='/cipd/login']", timeout=REQUEST_TIMEOUT)
            await page.wait_for_url(lambda url: "cipd.org" in url, timeout=REQUEST_TIMEOUT)

            # Fill in login form
            await page.fill("input[name='username']", USERNAME)
            await page.fill("input[name='password']", PASSWORD)
            await page.click("button[type='submit']")

            # Wait for redirect back to peoplemanagement
            await page.wait_for_url(lambda url: "peoplemanagement.co.uk" in url, timeout=REQUEST_TIMEOUT)

            print("Logged in successfully.")
            return True
        except Exception as e:
            print(f"Login attempt {attempt+1} failed: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                print("All login attempts failed.")
                return False

async def handle_cookie_banner(page):
    """Accepts the cookies if the banner is present"""
    try:
        # Try to locate the cookie button
        cookie_button = page.locator("#onetrust-accept-btn-handler")
        if await cookie_button.is_visible(timeout=3000):
            await cookie_button.click()
            print("Accepted all cookies.")
            await asyncio.sleep(0.5)  # Small delay after clicking
    except Exception:
        # Silently pass if not found or already accepted
        print("Cookie banner not found or already accepted.")

async def check_for_membership_lock(page):
    """Check if the page is locked for members only"""
    locked_selector = ".member-messaging__title:has-text('Members access only')"
    return await page.locator(locked_selector).count() > 0

async def extract_article_content(page, url):
    """Extract content from an article page with optimized selectors and error handling"""
    cached_data = await load_from_cache(url)
    if cached_data:
        print(f"Using cached data for: {url}")
        return cached_data

    article_data = {"url": url}

    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Use networkidle for initial load, then optimize for content extraction
            await page.goto(url, timeout=PAGE_NAVIGATION_TIMEOUT)
            await page.wait_for_load_state("domcontentloaded")
            
            # Add a small random delay to avoid detection
            await asyncio.sleep(random.uniform(0.5, 1.5))

            # Optimized extraction with error handling for each section
            # 1) Title - using waitForSelector to ensure element is loaded
            try:
                await page.wait_for_selector("h1[data-cy='articleHeading']", timeout=5000)
                title = await page.locator("h1[data-cy='articleHeading']").text_content()
                article_data["title"] = title.strip() if title else ""
            except Exception:
                article_data["title"] = ""

            # 2) Author
            try:
                author = await page.locator("span.authorName").text_content()
                article_data["author"] = author.replace("by", "").strip()
            except Exception:
                article_data["author"] = ""

            # 3) Date
            try:
                date_text = await page.locator("p.byline span:nth-child(2)").text_content()
                article_data["date"] = date_text.strip()

                # Try parsing year
                article_data["year"] = None
                try:
                    parsed_date = datetime.datetime.strptime(date_text.strip(), "%d %B %Y")
                    article_data["year"] = parsed_date.year
                except Exception:
                    pass
            except Exception:
                article_data["date"] = ""
                article_data["year"] = None

            try:
                # Target the specific p.summary element
                summary_p = page.locator("p.summary")
                print(f"Summary p count: {await summary_p.count()}")
                if await summary_p.count() > 0:
                    summary_text = await summary_p.text_content()
                    article_data["summary"] = summary_text.strip()
                else:
                    # Fallback options if p.summary isn't found
                    alternative_selectors = [
                        "summary",
                        "header em",
                        "p.gatedArticle__summary",
                        "section.page-intro p"
                    ]
                    
                    for selector in alternative_selectors:
                        selector_locator = page.locator(selector)
                        if await selector_locator.count() > 0:
                            text = await selector_locator.text_content()
                            if text and text.strip():
                                article_data["summary"] = text.strip()
                                break
            except Exception as e:
                print(f"Error extracting summary: {e}")
                article_data["summary"] = ""

            # 5) Description (use summary if no separate desc)
            article_data["description"] = []

            # 6) Tags & category
            article_data["tags"] = []
            article_data["category"] = ""

            # 7) Full Text - optimize to gather all text at once
            try:
                await page.wait_for_selector("#articleBody", timeout=5000)
                article_body = page.locator("#articleBody p")
                full_text_elements = await article_body.element_handles()
                
                full_text_lines = []
                for elem in full_text_elements:
                    text = await elem.text_content()
                    if text.strip():
                        full_text_lines.append(text.strip())
                
                article_data["full_text"] = unidecode("\n".join(full_text_lines))
            except Exception:
                article_data["full_text"] = ""

            # 8) Links - collect more efficiently
            try:
                link_elements = page.locator("#articleBody a")
                links = []
                for handle in await link_elements.element_handles():
                    href = await handle.get_attribute("href")
                    if href:
                        links.append(href)
                article_data["links"] = links
            except Exception:
                article_data["links"] = []

            # 9) Directory (breadcrumb) - not used, skip
            article_data["directory"] = ""

            # Successfully extracted content
            save_to_cache(url, article_data)
            return article_data
            
        except Exception as e:
            print(f"Error extracting content from {url} (attempt {attempt+1}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                # After all retries, return what we have
                article_data["full_text"] = article_data.get("full_text", "")
                article_data["links"] = article_data.get("links", [])
                return article_data

def parse_date(article):
    """Safely parse the date string in the article data for sorting purposes"""
    date_str = article.get("date", "01 Jan, 1900")
    try:
        # Try different date formats
        for fmt in ["%d %B %Y", "%d %b, %Y", "%B %d, %Y", "%d %b %Y"]:
            try:
                return datetime.datetime.strptime(date_str, fmt)
            except ValueError:
                continue
    except Exception:
        pass
    
    # Default fallback date
    return datetime.datetime(1900, 1, 1)

async def process_article(context, url, semaphore):
    """Process a single article with semaphore to limit concurrency"""
    async with semaphore:
        # Check cache first
        cached_data = await load_from_cache(url)
        if cached_data:
            print(f"Using cached data for: {url}")
            return cached_data

        # Create a new page for this article
        page = await context.new_page()
        try:
            result = await extract_article_content(page, url)
            return result
        finally:
            await page.close()

async def main_async():
    """Main function using async operations"""
    async with async_playwright() as p:
        # Use persistent context to improve performance
        browser_type = p.chromium
        browser = await browser_type.launch(headless=True)  # True for production

        context_options = {}
        if os.path.exists(STORAGE_STATE_PATH):
            context_options["storage_state"] = STORAGE_STATE_PATH
            print("Loaded existing storage state.")
        
        context = await browser.new_context(**context_options)
        
        # Set default timeouts
        context.set_default_timeout(REQUEST_TIMEOUT)
        
        try:
            # Initial login page
            page = await context.new_page()
            
            # Try loading home page
            await page.goto(HOME_URL)
            await page.wait_for_load_state("domcontentloaded")
            
            # Handle cookie banner and login
            #await handle_cookie_banner(page)
            #logged_in = await login(page)
            
            #if logged_in:
                # Save session after successful login
                #await context.storage_state(path=STORAGE_STATE_PATH)
            
            # Read JSON links
            with open(INPUT_JSON, "r") as f:
                scraped_data = json.load(f)
            
            # Get URLs to process (all or limit as needed)
            urls = scraped_data.get("articles", [])
            
            # Create a semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAGES)
            
            # Process articles concurrently with controlled parallelism
            tasks = [process_article(context, url, semaphore) for url in urls]
            articles = await asyncio.gather(*tasks)
            
            # Filter out None values (failed extractions)
            articles = [a for a in articles if a]
            
            # Sort and save
            articles.sort(key=lambda a: parse_date(a), reverse=True)
            
            with open(OUTPUT_JSON, "w") as out_f:
                json.dump({
                    "total_articles": len(articles),
                    "articles": articles
                }, out_f, indent=4)
            
            print(f"Extraction completed. Processed {len(articles)} articles. Data saved to {OUTPUT_JSON}")
        
        finally:
            # Save final state and close
            await context.storage_state(path=STORAGE_STATE_PATH)
            await browser.close()

def main():
    """Entry point that runs the async main function"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()