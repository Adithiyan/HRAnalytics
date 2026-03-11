import os
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
import time
# Load credentials from .env file
load_dotenv()
EMAIL = os.getenv("USERNAME")
PASSWORD = os.getenv("PASS")

print("EMAIL:", EMAIL)
print("PASSWORD:", PASSWORD)

# Define the URLs
LOGIN_URL = "https://login.shrm.org/?request_id=id0C9FEB4AECE60E&relay_state=id-aa032f59-b2bd-4169-8090-772550560b17&issuer=aHR0cHM6Ly9zc28uc2hybS5vcmcvSURCVVMvU0hSTS9JRFAvU0FNTDIvTUQ=&target=aHR0cHM6Ly9hZW0td3d3LnNoc2hybS5vcmcLw=="
NEWS_URL = "https://www.shrm.org/topics-tools/news"

# Set up Selenium WebDriver
driver = webdriver.Chrome()
driver.maximize_window()

try:
    # Open the login page
    driver.get(LOGIN_URL)

    # Wait until the email field is visible
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "LoginCredential_Email")))

    # Fill in the login form
    email_field = driver.find_element(By.ID, "LoginCredential_Email")
    email_field.send_keys(EMAIL)

    password_field = driver.find_element(By.ID, "LoginCredential_Password")
    password_field.send_keys(PASSWORD)

    # Submit the form
    password_field.send_keys(Keys.RETURN)

    # Wait for successful login
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))  # Adjust selector for post-login confirmation
    )

    print("Login successful.")

    # Navigate to the news page
    driver.get(NEWS_URL)
    time.sleep(250)

    # Wait for the page to load completely
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "result-card"))  # Adjust the selector for elements on the news page
    )

    

    print("Navigated to the news page.")

    # Accept cookies if the banner appears
    try:
        cookie_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Accept")]'))  # Adjust the XPATH as needed
        )
        cookie_button.click()
        print("Cookies accepted.")
    except Exception as e:
        print("No cookie banner found or already accepted.")

    # Save cookies to a JSON file
    cookies = driver.get_cookies()
    with open("shrm_cookies.json", "w") as file:
        json.dump(cookies, file, indent=4)

    print("Cookies saved successfully to shrm_cookies.json.")

finally:
    # Quit the WebDriver
    driver.quit()
