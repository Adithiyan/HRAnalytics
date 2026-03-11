"""
cipd/main_links.py — Stage 1: Login to CIPD and collect top-level category URLs.

Uses Selenium (Chrome) to:
  1. Log in at CIPD_LOGIN_URL with the configured credentials and persist
     the session as a pickle cookie file.
  2. Load the UK Policy & Insights hub and extract every 'a.link--arrow'
     href, saving the list to CIPD_MAIN_LINKS_FILE.

The login step opens a visible (non-headless) browser so the operator can
observe and handle any 2FA or CAPTCHA challenges.

Input:  None
Output: cipd/main_links_views_uk.txt  (one URL per line)
        cipd/cookies.pkl              (Selenium session cookies)

Usage:
    python -m cipd.main_links      # from src/
    python src/cipd/main_links.py  # from project root
"""

import os
import sys
import time
import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CIPD_USERNAME, CIPD_PASSWORD,
    CIPD_LOGIN_URL, CIPD_KNOWLEDGE_URL,
    CIPD_COOKIES_FILE, CIPD_MAIN_LINKS_FILE,
)


# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------

def save_cookies(driver, file_path):
    """Pickle the current browser session cookies to `file_path`."""
    with open(file_path, "wb") as file:
        pickle.dump(driver.get_cookies(), file)


def load_cookies(driver, file_path):
    """Restore cookies from a pickle file into the active Selenium session."""
    with open(file_path, "rb") as file:
        cookies = pickle.load(file)
        for cookie in cookies:
            driver.add_cookie(cookie)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def login_and_save_cookies():
    """
    Open a visible Chrome session, log in to CIPD, and pickle the cookies.

    The browser is left open for 5 s after form submission to allow
    the login redirect to complete before cookies are captured.
    """
    options = Options()
    # headless=False intentionally — operator may need to handle CAPTCHA/2FA
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(), options=options)

    try:
        driver.get(CIPD_LOGIN_URL)
        time.sleep(3)

        username_input = driver.find_element(By.NAME, "username")
        password_input = driver.find_element(By.NAME, "password")
        username_input.send_keys(CIPD_USERNAME)
        password_input.send_keys(CIPD_PASSWORD)
        password_input.send_keys(Keys.RETURN)

        time.sleep(5)
        save_cookies(driver, CIPD_COOKIES_FILE)
        print("Cookies saved successfully.")
    finally:
        driver.quit()


def extract_main_links():
    """
    Load CIPD's UK Policy & Insights hub with saved cookies and harvest
    all top-level 'a.link--arrow' hrefs, writing them to CIPD_MAIN_LINKS_FILE.
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(), options=options)

    try:
        driver.get(CIPD_KNOWLEDGE_URL)
        load_cookies(driver, CIPD_COOKIES_FILE)
        driver.refresh()
        time.sleep(3)

        main_links = []
        elements = driver.find_elements(By.CSS_SELECTOR, "a.link--arrow")
        for element in elements:
            href = element.get_attribute("href")
            if href:
                main_links.append(href)

        with open(CIPD_MAIN_LINKS_FILE, "w") as file:
            for link in main_links:
                file.write(link + "\n")

        print(f"Extracted {len(main_links)} main links and saved to {CIPD_MAIN_LINKS_FILE}.")
    finally:
        driver.quit()


def main():
    """Run login then main-link extraction sequentially."""
    login_and_save_cookies()
    extract_main_links()


if __name__ == "__main__":
    main()
