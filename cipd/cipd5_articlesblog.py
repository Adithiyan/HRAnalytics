
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
INPUT_JSON = "scraped_links_cipd_community_target.json" #"scraped_links_people_management.json"

OUTPUT_JSON = "cipd_article_links_people_management.json"
LOGIN_URL = "https://auth.cipd.org/u/login"
HOME_URL = "https://www.peoplemanagement.co.uk"
USERNAME = "Jade.yy423@gmail.com"
PASSWORD = "jadeCIPD123"

STORAGE_STATE_PATH = "storage_state.json"

def login(page):
    try:
        # Go to the homepage
        page.goto(HOME_URL)
        page.wait_for_load_state("networkidle")
        handle_cookie_banner(page)

        # Click the CIPD login link
        page.click("#myAccount a[href*='/cipd/login']", timeout=10000)
        page.wait_for_url(lambda url: "cipd.org" in url, timeout=10000)

        # Fill in login form
        page.fill("input[name='username']", USERNAME)
        page.fill("input[name='password']", PASSWORD)
        page.click("button[type='submit']")

        # Wait for redirect back to peoplemanagement
        page.wait_for_url(lambda url: "peoplemanagement.co.uk" in url, timeout=15000)

        print("Logged in successfully.")
    except Exception as e:
        print(f"Error during login: {e}")



def handle_cookie_banner(page):

    try:
        # Try to locate the cookie button
        if page.locator("#onetrust-accept-btn-handler").is_visible():
            page.click("#onetrust-accept-btn-handler", timeout=3000)
            print("Accepted all cookies.")
        else:
            print("No cookie banner shown.")
    except Exception:
        # Silently pass if not found or already accepted
        print("Cookie banner not found or already accepted.")


def check_for_membership_lock(page):

    locked_selector = ".member-messaging__title:has-text('Members access only')"
    return page.locator(locked_selector).count() > 0
def extract_article_content(page, url):
    article_data = {"url": url}

    try:
        page.goto(url, timeout=20000)
        #page.wait_for_load_state("networkidle")
        time.sleep(random.uniform(1, 3))

        # 1) Title
        try:
            title = page.locator("h1[data-cy='articleHeading']").text_content()
            article_data["title"] = title.strip() if title else ""
        except:
            article_data["title"] = ""
        print(f"Extracting content from: {url}")
        # 2) Author
        try:
            author = page.locator("span.authorName").text_content()
            article_data["author"] = author.replace("by", "").strip()
        except:
            article_data["author"] = ""

        # 3) Date
        try:
            date_text = page.locator("p.byline span:nth-child(2)").text_content().strip()
            article_data["date"] = date_text

            # Try parsing year
            article_data["year"] = None
            try:
                parsed_date = datetime.datetime.strptime(date_text, "%d %B %Y")
                article_data["year"] = parsed_date.year
            except:
                pass
        except:
            article_data["date"] = ""
            article_data["year"] = None

        # 4) Summary
        try:
            summary_em = page.locator("header em").text_content()
            article_data["summary"] = summary_em.strip()
        except:
            article_data["summary"] = ""

        # 5) Description
        article_data["description"] = article_data["summary"]  # use summary if no separate desc

        # 6) Tags & category – not clearly shown, so leave blank
        article_data["tags"] = []
        article_data["category"] = ""
        print(f"Extracted: {article_data['title']} by {article_data['author']} on {article_data['date']}")

        # 7) Full Text
        try:
            full_text_elements = page.locator("#articleBody p")
            full_text_lines = [elem.text_content().strip() for elem in full_text_elements.all() if elem.text_content().strip()]
            article_data["full_text"] = unidecode("\n".join(full_text_lines))
        except:
            article_data["full_text"] = ""
        try:
            summary_gated = page.locator("p.gatedArticle__summary").text_content()
            if summary_gated and summary_gated.strip():
                article_data["summary"] = summary_gated.strip()
            else:
                # fallback to previous summary extraction
                summary_locator = page.locator("section.page-intro")
                summary_texts = summary_locator.all_text_contents()
                summary_combined = " ".join(t.strip() for t in summary_texts if t.strip())
                article_data["summary"] = summary_combined.strip()
        except Exception:
            article_data["summary"] = ""
        # 8) Links
        try:
            link_elements = page.locator("#articleBody a")
            article_data["links"] = [a.get_attribute("href") for a in link_elements.element_handles() if a.get_attribute("href")]
        except:
            article_data["links"] = []

        # 9) Directory (breadcrumb) – not visibly used, skip
        article_data["directory"] = ""

    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        article_data["full_text"] = ""
        article_data["links"] = []
    print(f"✅ Extracted {len(article_data['full_text'])} chars, {len(article_data['links'])} links")
    return article_data


def parse_date(article):

    try:
        return datetime.datetime.strptime(article.get("date", "01 Jan, 1900"), "%d %b, %Y")
    except ValueError:
        return datetime.datetime(1900, 1, 1)

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        # Load session if exists; else create fresh
        if os.path.exists(STORAGE_STATE_PATH):
            context = browser.new_context(storage_state=STORAGE_STATE_PATH)
            print("Loaded existing storage state.")
        else:
            context = browser.new_context()
            print("No existing storage state. Creating a new session.")

        page = context.new_page()

        try:
            # 1) Go to homepage
            page.goto(HOME_URL)
            #page.wait_for_load_state("networkidle")
            page.wait_for_load_state("load")
            page.wait_for_timeout(2000) 


            # 2) Handle cookie banner
            #handle_cookie_banner(page)

            # 3) Login
            #login(page)

            # 4) Save session after login
            #context.storage_state(path=STORAGE_STATE_PATH)

            # 5) Read JSON links
            with open(INPUT_JSON, "r") as f:
                scraped_data = json.load(f)

            articles = []

            # 6) Limit to first 5 links
            urls = []
            with open(INPUT_JSON, "r") as f:
                scraped_data = json.load(f)
            urls = scraped_data.get("articles", [])[:10]  # correct

            for url in urls:
                print(f"Extracting content from: {url}")
                article_data = extract_article_content(page, url)
                articles.append(article_data)

            # 7) Sort and save
            articles.sort(key=lambda a: parse_date(a), reverse=True)

            with open(OUTPUT_JSON, "w") as out_f:
                json.dump({
                    "total_articles": len(articles),
                    "articles": articles
                }, out_f, indent=4)

            print(f"Extraction completed. Data saved to {OUTPUT_JSON}")

        finally:
            context.storage_state(path=STORAGE_STATE_PATH)
            browser.close()

if __name__ == "__main__":
    main()
