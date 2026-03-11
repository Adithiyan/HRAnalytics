import os
import json
import time
import random
from datetime import datetime
from playwright.sync_api import sync_playwright
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_text(text):
    if not text:
        return "No Data"
    return text.replace("opens in a new tab", "").strip().replace("\n", "").replace("\r", "")
def extract_article_content(url):
    content_data = {"url": url}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            logging.info(f"Extracting content from: {url}")
            page.goto(url, timeout=10000)
            page.wait_for_load_state("domcontentloaded")

            # Extract Title
            try:
                content_data["title"] = clean_text(page.text_content('h1.content__title', timeout=5000) or "No Title")
            except Exception as e:
                logging.warning(f"Title not found for {url}: {e}")
                content_data["title"] = "No Title"

            # Extract Category
            try:
                breadcrumb_items = page.query_selector_all(
                    'ol.cmp-breadcrumb__list li.cmp-breadcrumb__item span[itemprop="name"]')
                if len(breadcrumb_items) >= 2:
                    category = " > ".join([clean_text(item.text_content()) for item in breadcrumb_items[2:]])
                    content_data["category"] = category
                else:
                    content_data["category"] = "No Category"
            except Exception as e:
                logging.warning(f"Category not found for {url}: {e}")
                content_data["category"] = "No Category"

            # Extract Date
            try:
                date = clean_text(page.text_content('span.content__date', timeout=5000) or "No Date")
                content_data["date"] = date
                content_data["year"] = datetime.strptime(date, "%B %d, %Y").year
                content_data["datetime_obj"] = datetime.strptime(date, "%B %d, %Y")
            except Exception as e:
                logging.warning(f"Date not found or invalid for {url}: {e}")
                content_data["date"] = "No Date"
                content_data["year"] = "Unknown"
                content_data["datetime_obj"] = datetime.min

            # Extract Author
            try:
                content_data["author"] = clean_text(page.text_content('span.content__author', timeout=5000) or "No Author")
            except Exception as e:
                logging.warning(f"Author not found for {url}: {e}")
                content_data["author"] = "No Author"

            # Extract Type
            try:
                type_tag = page.query_selector('div.pretitle a[data-contentfiltertag]')
                content_data["type"] = clean_text(type_tag.text_content() if type_tag else "Unknown")
            except Exception as e:
                logging.warning(f"Type not found for {url}: {e}")
                content_data["type"] = "Unknown"

            # Extract Tags
            try:
                tags = page.query_selector_all('a[aria-label="button tag"]')
                tag_list = [clean_text(tag.text_content()) for tag in tags]
                content_data["tags"] = [tag for tag in tag_list if tag.lower() != content_data["type"].lower()]
            except Exception as e:
                logging.warning(f"Tags not found for {url}: {e}")
                content_data["tags"] = []

            # Extract Full Text or any available content
            try:
                main_content_div = page.query_selector('div.cmp-text')
                if main_content_div:
                    paragraphs = main_content_div.query_selector_all('p, h2, ul')
                    content_data["full_text"] = "\n\n".join(clean_text(p.text_content()) for p in paragraphs)
                else:
                    content_data["full_text"] = clean_text(page.content()) or "No Content"
            except Exception as e:
                logging.warning(f"Full text extraction failed for {url}: {e}")
                content_data["full_text"] = clean_text(page.content()) or "No Content"

            browser.close()

    except Exception as e:
        logging.error(f"Error extracting content from {url}: {e}")
        content_data["error"] = str(e)
        try:
            # Extract any remaining text content from the page
            content_data["fallback_content"] = clean_text(page.content())
        except Exception as e_inner:
            content_data["fallback_content"] = "No Additional Content"

    return content_data

def extract_article_content2(url):
    content_data = {"url": url}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            logging.info(f"Extracting content from: {url}")
            page.goto(url, timeout=10000)
            page.wait_for_load_state("domcontentloaded")

            # Extract Title
            content_data["title"] = clean_text(page.text_content('h1.content__title', timeout=5000) or "No Title")

            # Extract Category
            breadcrumb_items = page.query_selector_all(
                'ol.cmp-breadcrumb__list li.cmp-breadcrumb__item span[itemprop="name"]')
            if len(breadcrumb_items) >= 2:
                category = " > ".join([clean_text(item.text_content()) for item in breadcrumb_items[2:]])
                content_data["category"] = category
            else:
                content_data["category"] = "No Category"

            # Extract Date
            date = clean_text(page.text_content('span.content__date', timeout=5000) or "No Date")
            content_data["date"] = date
            try:
                content_data["year"] = datetime.strptime(date, "%B %d, %Y").year
                content_data["datetime_obj"] = datetime.strptime(date, "%B %d, %Y")
            except ValueError:
                content_data["year"] = "Unknown"
                content_data["datetime_obj"] = datetime.min

            # Extract Author
            content_data["author"] = clean_text(page.text_content('span.content__author', timeout=5000) or "No Author")

            # Extract Type
            type_tag = page.query_selector('div.pretitle a[data-contentfiltertag]')
            content_data["type"] = clean_text(type_tag.text_content() if type_tag else "Unknown")

            # Extract Tags
            tags = page.query_selector_all('a[aria-label="button tag"]')
            tag_list = [clean_text(tag.text_content()) for tag in tags]
            content_data["tags"] = [tag for tag in tag_list if tag.lower() != content_data["type"].lower()]

            # Extract Full Text
            main_content_div = page.query_selector('div.cmp-text')
            if main_content_div:
                paragraphs = main_content_div.query_selector_all('p, h2, ul')
                content_data["full_text"] = "\n\n".join(clean_text(p.text_content()) for p in paragraphs)
            else:
                content_data["full_text"] = "No Content"

            browser.close()

    except Exception as e:
        logging.error(f"Error extracting content from {url}: {e}")
        content_data["error"] = str(e)

    return content_data

def save_to_json(data, file_name):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logging.info(f"Data saved to {file_path}")

def process_links(links, output_file):
    all_articles = []
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust the number of workers as needed
            future_to_url = {executor.submit(extract_article_content, link): link for link in links}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    article_content = future.result()
                    all_articles.append(article_content)
                    
                    # Convert all datetime_obj to datetime for sorting
                    for article in all_articles:
                        if "datetime_obj" in article:
                            if isinstance(article["datetime_obj"], str):
                                try:
                                    article["datetime_obj"] = datetime.fromisoformat(article["datetime_obj"])
                                except ValueError:
                                    article["datetime_obj"] = datetime.min
                    
                    # Sort articles by date (most recent first)
                    sorted_articles = sorted(
                        all_articles,
                        key=lambda x: x.get("datetime_obj", datetime.min),
                        reverse=True
                    )
                    
                    # Convert datetime objects to strings for JSON serialization
                    for article in sorted_articles:
                        if "datetime_obj" in article and isinstance(article["datetime_obj"], datetime):
                            article["datetime_obj"] = article["datetime_obj"].isoformat()
                    
                    # Add total articles count at the beginning
                    output_data = {
                        "total_articles": len(sorted_articles),
                        "articles": sorted_articles
                    }

                    # Save progress after every article
                    save_to_json(output_data, output_file)

                except Exception as e:
                    logging.error(f"Error processing link {url}: {e}")

    except Exception as e:
        logging.critical(f"Critical error during processing: {e}")
    
    return all_articles



if __name__ == "__main__":
    input_file = "output/shrm_article_links_news20250119_024823.txt"#"output\shrm_article_links_non_news20250119_020331.txt"#"output/article_links_20250102_005320.txt"
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            links = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"File {input_file} not found. Ensure the file exists.")
        exit(1)

    output_file = f"articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    logging.info("Starting article extraction process...")

    process_links(links, output_file)

    logging.info("Processing complete. All articles saved.")
