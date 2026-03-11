import os
import json
import time
import random
from datetime import datetime
from playwright.sync_api import sync_playwright

def clean_text(text):
    if not text:
        return "No Data"
    return text.replace("opens in a new tab", "").strip().replace("\n", "").replace("\r", "")

def extract_article_content(url):
    content_data = {"url": url}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            print(f"Extracting content from: {url}")
            page.goto(url)
            page.wait_for_load_state("domcontentloaded", timeout=10000)

            # Extract and clean Title
            try:
                title = page.text_content('h1.content__title', timeout=5000)
                content_data["title"] = clean_text(title)
            except Exception as e:
                content_data["title"] = "No Title"

            # Extract Category from Breadcrumb
            breadcrumb_items = page.query_selector_all('ol.cmp-breadcrumb__list li.cmp-breadcrumb__item span[itemprop="name"]')
            if len(breadcrumb_items) >= 2:
                category = " > ".join([clean_text(breadcrumb_items[i].text_content()) for i in range(2)])
                content_data["category"] = clean_text(category)
            else:
                content_data["category"] = "No Category"

            # Extract and clean Date
            try:
                date = page.text_content('span.content__date', timeout=5000)
                content_data["date"] = clean_text(date)

                # Extract Year from Date
                content_data["year"] = datetime.strptime(content_data["date"], "%B %d, %Y").year
            except Exception as e:
                content_data["date"] = "No Date"
                content_data["year"] = "Unknown"

            # Extract Author
            try:
                author = page.text_content('span.content__author', timeout=5000)
                content_data["author"] = clean_text(author)
            except Exception as e:
                content_data["author"] = "No Author"

            # Extract Type
            try:
                type_tag = page.query_selector('div.pretitle a[data-contentfiltertag]')
                if type_tag:
                    content_data["type"] = clean_text(type_tag.text_content())
                else:
                    content_data["type"] = "Unknown"
            except Exception as e:
                content_data["type"] = "Unknown"

            # Extract Tags and Remove Type from Tags
            tags = page.query_selector_all('a[aria-label="button tag"]')
            tag_list = [clean_text(tag.text_content()) for tag in tags]

            # Exclude Type from Tags
            content_data["tags"] = [tag for tag in tag_list if tag.lower() != content_data["type"].lower()]

            # Extract Full Text
            try:
                main_content_div = page.query_selector('div.cmp-text')
                if main_content_div:
                    paragraphs = main_content_div.query_selector_all('p, h2, ul')
                    content_data["full_text"] = "\n\n".join(clean_text(p.text_content()) for p in paragraphs)
                else:
                    content_data["full_text"] = "No Content"
            except Exception as e:
                content_data["full_text"] = "No Content"

        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            content_data["error"] = str(e)

            # Attempt to extract any remaining text content
            try:
                page_content = page.content()
                content_data["page_content"] = clean_text(page_content)
            except Exception as e:
                content_data["page_content"] = "No Additional Content"

        finally:
            browser.close()

    return content_data

def save_to_json(data, file_name):
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    input_file = "output/article_links_20250101_181635.txt"
    #input_file = "article_links.txt"  # Replace with your input file name
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            links = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"File {input_file} not found. Ensure the file exists.")
        exit(1)

    all_articles = []
    output_file = f"articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    for idx, link in enumerate(links):
        try:
            print(f"Processing article {idx + 1}/{len(links)}")
            time.sleep(random.uniform(1, 2))  # Random delay to avoid getting blocked
            article_content = extract_article_content(link)
            all_articles.append(article_content)

            # Save progress after every article
            save_to_json(all_articles, output_file)

        except Exception as e:
            print(f"Error processing link {link}: {e}")

    print("Processing complete. All articles saved.")
