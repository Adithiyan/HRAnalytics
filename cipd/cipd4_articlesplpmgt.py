import os
import json
import time
import datetime
import re
from unidecode import unidecode
from playwright.sync_api import sync_playwright

# --- Config ---
INPUT_JSON = "scraped_links_cipd_community.json"
OUTPUT_JSON = "cipd_article_data_community.json"
BASE_URL = "https://community.cipd.co.uk"

def extract_blog_article(page, url):
    article = {"url": url}

    try:
        page.goto(url, timeout=20000)
        page.wait_for_load_state("networkidle")
        time.sleep(1.5)

        # Title
        try:
            title = page.locator("h3.name").text_content().strip()
            article["title"] = title
        except:
            article["title"] = ""

        # Date & Author (combined in one <p>)
        try:
            author_block = page.locator("p:has(.author)").first.text_content().strip()
            match = re.search(r"By\s+(.*?)\s+(\d{1,2}\s\w{3},\s\d{4})", author_block)
            if match:
                article["author"] = match.group(1).strip()
                article["date"] = match.group(2).strip()
            else:
                article["author"] = ""
                article["date"] = ""
        except:
            article["author"] = ""
            article["date"] = ""

        # Tags
        try:
            tags = page.locator("ul.tag-list li").all_inner_texts()
            article["tags"] = [t.strip() for t in tags if t.strip()]
        except:
            article["tags"] = []

        # Full text
        try:
            content_blocks = page.locator("div.content > p").all_inner_texts()
            full_text = "\n\n".join([unidecode(p.strip()) for p in content_blocks if p.strip()])
            article["full_text"] = full_text
        except:
            article["full_text"] = ""

    except Exception as e:
        print(f"⚠️ Error scraping {url}: {e}")
        article["full_text"] = ""

    return article

def parse_date_safe(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%d %b, %Y")
    except:
        return datetime.datetime(1900, 1, 1)

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Read previously scraped blog links
        with open(INPUT_JSON, "r") as f:
            link_data = json.load(f)

        all_articles = []
        for url in link_data.get("articles", []):
            print(f"📄 Scraping: {url}")
            article = extract_blog_article(page, url)
            all_articles.append(article)

        # Sort by date (if available)
        all_articles.sort(key=lambda a: parse_date_safe(a.get("date", "")), reverse=True)

        # Save output
        with open(OUTPUT_JSON, "w") as f:
            json.dump({
                "total_articles": len(all_articles),
                "articles": all_articles
            }, f, indent=4)

        print(f"\n✅ Finished! Saved {len(all_articles)} articles to {OUTPUT_JSON}")

        browser.close()

if __name__ == "__main__":
    main()
