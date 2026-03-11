from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json

# --- Config ---
BASE_URL = "https://www.peoplemanagement.co.uk"
START_URL = BASE_URL + "/search/articles"
OUTPUT_JSON = "scraped_links_people_management.json"

# --- Setup WebDriver ---
def setup_driver():
    options = Options()
    # options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    return webdriver.Chrome(service=Service(), options=options)

# --- Accept cookies (run only once) ---
def accept_cookies_once(driver):
    try:
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        ).click()
        print("✅ Cookies accepted.")
    except:
        print("⚠️ Cookie popup not present or already accepted.")

# --- Extract article links ---
def extract_article_links(driver):
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.searchItem.storyContent"))
        )
        containers = driver.find_elements(By.CSS_SELECTOR, "div.searchItem.storyContent")

        links = []
        for container in containers:
            try:
                link_tag = container.find_element(By.CSS_SELECTOR, "h1 a")
                href = link_tag.get_attribute("href")
                if href and href.startswith("/"):
                    href = BASE_URL + href
                links.append(href)
            except:
                continue
        return links
    except Exception as e:
        print(f"⚠️ Failed to extract articles: {e}")
        return []

# --- Find next page URL ---
def get_next_page_url(driver):
    try:
        next_btn = driver.find_element(By.CSS_SELECTOR, "div.paginationNext a")
        href = next_btn.get_attribute("href")
        if href and href.startswith("/"):
            return BASE_URL + href
        return href
    except:
        return None

# --- Main scrape routine ---
def scrape_articles():
    driver = setup_driver()
    all_links = []
    current_url = START_URL
    is_first_page = True

    try:
        while current_url:
            print(f"\n🔍 Visiting: {current_url}")
            driver.get(current_url)
            time.sleep(2)

            if is_first_page:
                accept_cookies_once(driver)
                is_first_page = False

            links = extract_article_links(driver)
            print(f"📄 {len(links)} links found on this page.")
            all_links.extend(links)

            next_url = get_next_page_url(driver)
            if next_url and next_url != current_url:
                current_url = next_url
            else:
                print("🚫 No more pages.")
                break

        unique_links = list(set(all_links))
        with open(OUTPUT_JSON, "w") as f:
            json.dump({
                "start_url": START_URL,
                "total_articles": len(unique_links),
                "articles": unique_links
            }, f, indent=4)

        print(f"\n✅ Finished. {len(unique_links)} unique articles saved to {OUTPUT_JSON}")

    finally:
        driver.quit()

# --- Entry point ---
if __name__ == "__main__":
    scrape_articles()
