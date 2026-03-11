from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import pickle

# Configuration
#LOGIN_URL = "https://www.cipd.org/login"
#KNOWLEDGE_URL = "https://www.cipd.org/en/views-and-insights/"
LOGIN_URL = "https://www.cipd.org/login" #UK
KNOWLEDGE_URL = "https://www.cipd.org/uk/policy-and-insights/" #UK
COOKIES_FILE = "cookies.pkl"
MAIN_LINKS_FILE = "main_links_views_uk.txt"

# Credentials (Replace with secure method to fetch credentials)
USERNAME = "Jade.yy423@gmail.com"
PASSWORD = "jadeCIPD123"

def save_cookies(driver, file_path):
    """Save cookies to a file."""
    with open(file_path, "wb") as file:
        pickle.dump(driver.get_cookies(), file)

def load_cookies(driver, file_path):
    """Load cookies from a file."""
    with open(file_path, "rb") as file:
        cookies = pickle.load(file)
        for cookie in cookies:
            driver.add_cookie(cookie)

def login_and_save_cookies():
    """Logs in using Selenium and saves the cookies."""
    options = Options()
    #options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(), options=options)

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

def extract_main_links():
    """Extracts main links from the knowledge page and saves them to a file."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(), options=options)

    try:
        driver.get(KNOWLEDGE_URL)

        # Load cookies
        load_cookies(driver, COOKIES_FILE)
        driver.refresh()
        time.sleep(3)  # Wait for cookies to take effect

        # Extract main links
        main_links = []
        elements = driver.find_elements(By.CSS_SELECTOR, "a.link--arrow")
        for element in elements:
            href = element.get_attribute("href")
            if href:
                main_links.append(href)

        # Save main links to a file
        with open(MAIN_LINKS_FILE, "w") as file:
            for link in main_links:
                file.write(link + "\n")

        print(f"Extracted {len(main_links)} main links and saved to {MAIN_LINKS_FILE}.")
    finally:
        driver.quit()

def main():
    """Main function to log in and extract main links."""
    login_and_save_cookies()
    extract_main_links()

if __name__ == "__main__":
    main()
