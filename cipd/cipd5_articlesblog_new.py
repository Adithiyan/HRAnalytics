import json
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
import time
import random

INPUT_JSON = "scraped_links_cipd_community_target.json"
OUTPUT_JSON = "cipd_articles_static_scrape.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def debug_page_structure(soup, url):
    """Debug function to understand page structure"""
    print(f"🔍 Debugging page structure for: {url}")
    
    # Check for common content containers
    selectors_to_check = [
        "div.content.full > div.content",
        "div.content.full",
        "div.content",
        ".content",
        "article",
        ".post-content",
        ".blog-post",
        "main",
        "#main-content"
    ]
    
    for selector in selectors_to_check:
        elements = soup.select(selector)
        print(f"  '{selector}': {len(elements)} elements found")
        if elements:
            # Show first few characters of text content
            text_preview = elements[0].get_text(strip=True)[:100]
            print(f"    Preview: {text_preview}...")
    
    # Check for title elements
    title_selectors = [
        "div.content.full > h3.name",
        "h1",
        "h2",
        "h3",
        ".title",
        ".post-title"
    ]
    
    print("📝 Title candidates:")
    for selector in title_selectors:
        elements = soup.select(selector)
        for i, elem in enumerate(elements[:3]):  # Show first 3 matches
            print(f"  '{selector}' [{i}]: {elem.get_text(strip=True)[:50]}...")

def extract_article_content(url):
    print(f"\n🔗 Fetching: {url}")
    article = {"url": url}

    try:
        # Add random delay to be respectful
        time.sleep(random.uniform(1, 2))
        
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Debug page structure if needed
        # debug_page_structure(soup, url)

        # Try multiple selectors for title
        title = ""
        title_selectors = [
            "div.content.full > h3.name",
            "h1",
            "h2.post-title",
            ".post-title",
            "h3.name",
            "title"
        ]
        
        for selector in title_selectors:
            title_el = soup.select_one(selector)
            if title_el and title_el.get_text(strip=True):
                title = title_el.get_text(strip=True)
                print(f"📝 Title found with '{selector}': {title[:50]}...")
                break
        
        article["title"] = title

        # Try multiple selectors for main content
        main_content = None
        content_selectors = [
            "div.content.full > div.content",
            "div.content.full",
            "div.content",
            ".content",
            "article",
            ".post-content",
            ".blog-content",
            "main"
        ]
        
        for selector in content_selectors:
            content_candidate = soup.select_one(selector)
            if content_candidate:
                main_content = content_candidate
                print(f"📄 Content found with '{selector}'")
                break
        
        if not main_content:
            print("❌ No main content container found")
            # Fallback: try to get all paragraphs from body
            main_content = soup.find("body")
            if not main_content:
                raise Exception("No content container found")

        # Extract author
        author = ""
        if main_content:
            # Try different author patterns
            author_patterns = [
                lambda content: content.find("p", string=lambda text: text and "By " in text),
                lambda content: content.find("p", lambda tag: tag.find("strong") and "By " in tag.get_text()),
                lambda content: content.find(class_="author"),
                lambda content: content.find(class_="by-author"),
                lambda content: content.select_one(".author-name")
            ]
            
            for pattern in author_patterns:
                try:
                    author_elem = pattern(main_content)
                    if author_elem:
                        author_text = author_elem.get_text(strip=True)
                        if "By " in author_text:
                            author = author_text.replace("By ", "").strip()
                        else:
                            author = author_text.strip()
                        author_elem.decompose()  # Remove from content
                        break
                except:
                    continue

        article["author"] = author

        # Extract full text
        full_text = ""
        if main_content:
            # Get all paragraphs
            paragraphs = main_content.find_all("p")
            if paragraphs:
                paragraph_texts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 10:  # Filter out very short paragraphs
                        paragraph_texts.append(text)
                
                full_text = "\n\n".join(paragraph_texts)
            else:
                # Fallback: get all text from content
                full_text = main_content.get_text(separator="\n", strip=True)
            
            # Clean up text
            if full_text:
                full_text = unidecode(full_text)
                # Remove excessive whitespace
                full_text = "\n".join(line.strip() for line in full_text.split("\n") if line.strip())

        article["full_text"] = full_text

        # Extract links
        links = []
        if main_content:
            link_elements = main_content.find_all("a", href=True)
            for a in link_elements:
                href = a.get("href", "").strip()
                if href and not href.startswith("#"):  # Skip anchor links
                    # Convert relative URLs to absolute
                    if href.startswith("/"):
                        href = "https://community.cipd.co.uk" + href
                    links.append(href)
        
        article["links"] = list(set(links))  # Remove duplicates

        # Generate metadata
        summary = full_text[:250].strip() if full_text else ""
        if len(full_text) > 250:
            summary += "..."
        
        article["summary"] = summary
        article["tags"] = []
        article["category"] = ""
        article["directory"] = ""
                # Extract directory/breadcrumb information
        directory = ""
        directory_container = soup.select_one("div.row ul")
        if directory_container:
            breadcrumb_items = []
            for li in directory_container.find_all("li"):
                # Get text content, preferring link text over plain text
                link = li.find("a")
                if link and link.get_text(strip=True):
                    breadcrumb_items.append(link.get_text(strip=True))
                elif li.get_text(strip=True):
                    breadcrumb_items.append(li.get_text(strip=True))
            
            # Join breadcrumb items, excluding "Home"
            if breadcrumb_items:
                breadcrumb_items = [item for item in breadcrumb_items if item.lower() != "home"]
                directory = " > ".join(breadcrumb_items)
                print(f"📂 Directory found: {directory}")
        
        article["directory"] = directory

        print(f"✅ Extracted: '{title[:40]}...', {len(full_text)} chars, {len(links)} links")
        
        # Show some content preview for verification
        if full_text:
            preview = full_text[:150].replace("\n", " ")
            print(f"📖 Content preview: {preview}...")

    except Exception as e:
        print(f"❌ Failed to fetch {url}: {str(e)}")
        # Still save the URL for reference
        article.update({
            "title": "",
            "author": "",
            "full_text": "",
            "summary": "",
            "description": "",
            "links": [],
            "tags": [],
            "category": "",
            "directory": "",
            "error": str(e)
        })

    return article

def main():
    print("🚀 Starting CIPD article scraper...")
    
    try:
        with open(INPUT_JSON, "r") as f:
            url_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Input file '{INPUT_JSON}' not found!")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in '{INPUT_JSON}'!")
        return
    
    urls = url_data.get("articles", [])
    
    if not urls:
        print("❌ No URLs found in the input file!")
        return
    
    # Limit to first 10 for testing (remove [:10] for all URLs)
    
    
    print(f"\n📄 Processing {len(urls)} URLs...")
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n--- [{i}/{len(urls)}] ---")
        article = extract_article_content(url)
        results.append(article)
        
        # Show progress
        if i % 5 == 0:
            successful = sum(1 for r in results if r.get("full_text"))
            print(f"📊 Progress: {i}/{len(urls)} processed, {successful} successful extractions")

    # Save results
    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump({
                "total_articles": len(results),
                "successful_extractions": sum(1 for r in results if r.get("full_text")),
                "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "articles": results
            }, f, indent=2, ensure_ascii=False)
        
        successful = sum(1 for r in results if r.get("full_text"))
        print(f"\n✅ Done! Results saved to '{OUTPUT_JSON}'")
        print(f"📊 Summary: {successful}/{len(results)} articles successfully extracted")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

if __name__ == "__main__":
    main()