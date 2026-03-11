
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json

# --- Config ---
BASE_URL = "https://community.cipd.co.uk"
BLOG_URL_TEMPLATE = "https://community.cipd.co.uk/cipd-blogs"
OUTPUT_JSON = "scraped_links_cipd_community.json"
TOTAL_PAGES = 26

# --- Setup WebDriver ---
def setup_driver():
    options = Options()
    # options.add_argument("--headless")  # Uncomment to run headless
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    return webdriver.Chrome(service=Service(), options=options)

# --- Accept cookies once ---
def accept_cookies_once(driver):
    try:
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        ).click()
        print("✅ Cookies accepted.")
    except:
        print("⚠️ No cookie popup or already handled.")

# --- Extract article links from page ---
def extract_blog_links(driver):
    links = []
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.col-md-15.blog-post"))
        )
        posts = driver.find_elements(By.CSS_SELECTOR, "div.col-md-15.blog-post")

        for post in posts:
            try:
                link_tag = post.find_element(By.CSS_SELECTOR, "h4 a")
                href = link_tag.get_attribute("href")
                if href:
                    links.append(href)
            except:
                continue
    except Exception as e:
        print(f"⚠️ Error loading posts: {e}")
    return links
def scrape_cipd_blog_articles():
    driver = setup_driver()
    all_links = []
    is_first_page = True

    try:
        for page in range(1, TOTAL_PAGES + 1):
            url = f"{BLOG_URL_TEMPLATE}?pifragment-36={page}"
            print(f"\n🔍 Visiting page {page}: {url}")
            driver.get(url)
            time.sleep(2)  # wait for page to load

            if is_first_page:
                accept_cookies_once(driver)
                is_first_page = False

            print(f"⏳ Extracting blog links from page {page}...")
            page_links = extract_blog_links(driver)
            print(f"✅ Found {len(page_links)} blog links on page {page}")
            all_links.extend(page_links)

        # Remove duplicates
        unique_links = list(set(all_links))
        print(f"\n🧹 Removing duplicates... Final unique blog links: {len(unique_links)}")

        result = {
            "total_articles": len(unique_links),
            "articles": unique_links
        }

        with open(OUTPUT_JSON, "w") as f:
            json.dump(result, f, indent=4)

        print(f"\n💾 Saved to {OUTPUT_JSON}")

    except Exception as e:
        print(f"❌ Unexpected error during scraping: {e}")
    finally:
        driver.quit()
        print("🚪 Browser closed.")

# --- Main scrape function ---
def scrape_cipd_blog_articles2():
    driver = setup_driver()
    all_links = []
    is_first_page = True

    try:
        for page in range(1, TOTAL_PAGES + 1):
            url = BLOG_URL_TEMPLATE.format(page=page)
            print(f"\n🔍 Visiting: {url}")
            driver.get(url)
            time.sleep(2)

            if is_first_page:
                accept_cookies_once(driver)
                is_first_page = False

            page_links = extract_blog_links(driver)
            print(f"📄 Found {len(page_links)} links on page {page}")
            all_links.extend(page_links)

        unique_links = list(set(all_links))
        result = {
            "total_articles": len(unique_links),
            "articles": unique_links
        }

        with open(OUTPUT_JSON, "w") as f:
            json.dump(result, f, indent=4)

        print(f"\n✅ Done! {len(unique_links)} unique blog links saved to {OUTPUT_JSON}")

    finally:
        driver.quit()

# --- Entry ---
if __name__ == "__main__":
    scrape_cipd_blog_articles()
