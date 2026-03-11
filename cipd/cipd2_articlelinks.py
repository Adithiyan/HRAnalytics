from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import pickle
import json
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration
LOGIN_URL = "https://www.cipd.org/login"
COOKIES_FILE = "cookies.pkl"
MAIN_LINKS_FILE = "main_links_views_uk.txt"
OUTPUT_JSON = "scraped_links_views_uk.json"                                  
USERNAME = "Jade.yy423@gmail.com"
PASSWORD = "jadeCIPD123"

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def handle_cookie_banner(driver):
    """Handles the cookie banner by rejecting all cookies."""
    try:
        # Wait for the Cookie Settings button to be visible
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "onetrust-pc-btn-handler"))
        ).click()

        print("Clicked on 'Cookie Settings'.")

        # Wait for the Reject All button to appear and click it
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "ot-pc-refuse-all-handler"))
        ).click()

        print("Rejected all cookies.")
    except Exception as e:
        print(f"Error handling cookie banner: {e}")

# Set up Selenium options
def setup_driver():
    options = Options()
    #options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    return webdriver.Chrome(service=Service(), options=options)

def save_cookies(driver, file_path):
    """Save cookies to a file."""
    with open(file_path, "wb") as file:
        pickle.dump(driver.get_cookies(), file)

def load_cookies(driver, file_path, domain_url):
    """Load cookies from a file for the specific domain."""
    with open(file_path, "rb") as file:
        cookies = pickle.load(file)
        for cookie in cookies:
            if "domain" in cookie and domain_url in cookie["domain"]:
                driver.add_cookie(cookie)

def login_and_save_cookies():
    """Logs in using Selenium and saves the cookies."""
    driver = setup_driver()
    try:
        driver.get(LOGIN_URL)
        time.sleep(3)

        # Fill in login form
        username_input = driver.find_element(By.NAME, "username")
        password_input = driver.find_element(By.NAME, "password")
        username_input.send_keys(USERNAME)
        password_input.send_keys(PASSWORD)
        password_input.send_keys(Keys.RETURN)

        time.sleep(5)  # Wait for login to complete
        save_cookies(driver, COOKIES_FILE)
        print("Cookies saved successfully.")
    finally:
        driver.quit()

def extract_article_links(driver, url):
    """Extract article links from a specific section on the page."""
    driver.get(url)
    time.sleep(3)

    article_links = []
    cards = driver.find_elements(By.CSS_SELECTOR, "div.card.card--full")
    for card in cards:
        try:
            link_element = card.find_element(By.CSS_SELECTOR, "a.link--arrow")
            href = link_element.get_attribute("href")
            if href:
                article_links.append(href)
        except Exception as e:
            print(f"Error extracting link from card: {e}")

    return article_links

def scrape_main_links_with_pagination():
    """Scrapes all article links from the main links in the text file, handling pagination."""
    driver = setup_driver()

    try:
        # Load the main CIPD domain to set cookies
        driver.get("https://www.cipd.org/")
        time.sleep(3)  # Allow the page to load

        # Handle the cookie banner
        handle_cookie_banner(driver)

        # Load cookies
        load_cookies(driver, COOKIES_FILE, "cipd.org")
        driver.refresh()
        time.sleep(3)  # Wait for cookies to take effect

        # Load main links from the file
        with open(MAIN_LINKS_FILE, "r") as file:
            main_links = [line.strip() for line in file.readlines()]

        scraped_data = {}

        for main_link in main_links:
            print(f"Scraping: {main_link}")

            driver.get(main_link)
            time.sleep(3)

            # Handle the cookie banner again if it reappears
            handle_cookie_banner(driver)

            all_links = []
            while True:
                # Extract article links from the current page
                try:
                    article_links = extract_article_links(driver, driver.current_url)
                    all_links.extend(article_links)
                    print(f"Collected {len(article_links)} articles from this page.")
                except Exception as e:
                    print(f"Error extracting articles: {e}")
                    break

                # Check for pagination
                try:
                    wait = WebDriverWait(driver, 10)
                    next_button = wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.pagination__dir--next"))
                    )

                    # Scroll into view to ensure it's interactable
                    driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                    time.sleep(1)  # Small delay to stabilize the DOM

                    # Attempt to click the button
                    ActionChains(driver).move_to_element(next_button).click(next_button).perform()
                    time.sleep(3)  # Wait for the next page to load

                except Exception as e:
                    print(f"No more pages or error navigating to next page: {e}")
                    break

            # Add scraped links and count to the main link entry
            scraped_data[main_link] = {
                "articles": all_links,
                "total_articles": len(all_links)
            }

        # Save to JSON file
        with open(OUTPUT_JSON, "w") as json_file:
            json.dump(scraped_data, json_file, indent=4)

        print(f"Scraping completed. Data saved to {OUTPUT_JSON}")
    finally:
        driver.quit()

def main():
    print("Starting login and cookie saving process...")
    login_and_save_cookies()
    print("Starting scraping process...")
    scrape_main_links_with_pagination()

if __name__ == "__main__":
    main()
