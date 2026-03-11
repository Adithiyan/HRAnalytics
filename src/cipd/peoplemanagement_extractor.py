"""
cipd/peoplemanagement_extractor.py - Extract People Management articles with resume.
"""

import datetime as dt
import json
import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover - dependency may be absent in lean envs
    def unidecode(text):
        return text

from config import CIPD_PEOPLE_DATA_FILE, CIPD_PEOPLE_LINKS_FILE, PROJECT_ROOT

BASE_URL = "https://www.peoplemanagement.co.uk"
CHECKPOINT_FILE = os.path.join(PROJECT_ROOT, "src", "checkpoints", "cipd_people_articles.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _parse_date(text):
    if not text:
        return "", None
    text = re.sub(r"\s+", " ", text).strip()
    for fmt in ("%d %B %Y", "%d %b %Y", "%d %b, %Y"):
        try:
            parsed = dt.datetime.strptime(text, fmt)
            return parsed.strftime("%d %b %Y"), parsed.year
        except ValueError:
            pass
    return text, None


def _extract_date_and_year(soup):
    # Primary target.
    date_nodes = soup.select("p.byline span")
    for node in date_nodes:
        value = node.get_text(" ", strip=True)
        if re.search(r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}", value):
            return _parse_date(value)

    # Fallback.
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        text = time_tag.get_text(" ", strip=True)
        date_text, year = _parse_date(text)
        if year is None:
            raw = time_tag.get("datetime", "")
            if len(raw) >= 4 and raw[:4].isdigit():
                year = int(raw[:4])
        return date_text, year

    return "", None


def extract_article(url, dry_run=False):
    if dry_run:
        return {
            "url": url,
            "title": "Mock People Management Article",
            "author": "Mock Author",
            "date": "01 Jan 2025",
            "year": 2025,
            "category": "",
            "tags": [],
            "description": "Mock description",
            "summary": "Mock summary",
            "full_text": "Mock content",
            "links": [],
            "directory": "People Management",
        }

    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title = ""
    for sel in ["h1[data-cy='articleHeading']", "h1", "title"]:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            title = el.get_text(" ", strip=True)
            break

    author = ""
    for sel in ["span.authorName", "meta[name='author']"]:
        el = soup.select_one(sel)
        if not el:
            continue
        if el.name == "meta":
            val = el.get("content", "").strip()
        else:
            val = el.get_text(" ", strip=True)
        if val:
            author = val.replace("by ", "").replace("By ", "").strip()
            break

    date_text, year = _extract_date_and_year(soup)

    summary = ""
    for sel in ["p.summary", "p.gatedArticle__summary", "header em", "meta[name='description']"]:
        el = soup.select_one(sel)
        if not el:
            continue
        if el.name == "meta":
            val = el.get("content", "").strip()
        else:
            val = el.get_text(" ", strip=True)
        if val:
            summary = unidecode(val)
            break

    body = soup.select_one("#articleBody") or soup.select_one("article") or soup.body
    paragraphs = []
    links = []
    if body:
        for p in body.find_all("p"):
            txt = p.get_text(" ", strip=True)
            if txt:
                paragraphs.append(unidecode(txt))
        for a in body.find_all("a", href=True):
            href = a.get("href", "").strip()
            if href and not href.startswith("#"):
                links.append(urljoin(BASE_URL, href))

    full_text = "\n".join(paragraphs).strip()
    if not summary:
        summary = (full_text[:300] + "...") if len(full_text) > 300 else full_text

    breadcrumb = [x.get_text(" ", strip=True) for x in soup.select("ol.breadcrumb li") if x.get_text(strip=True)]
    directory = " > ".join(breadcrumb)

    tags = [t.get_text(" ", strip=True) for t in soup.select(".tag, .tags a") if t.get_text(strip=True)]

    return {
        "url": url,
        "title": title,
        "author": author,
        "date": date_text,
        "year": year,
        "category": "",
        "tags": sorted(set(tags)),
        "description": summary,
        "summary": summary,
        "full_text": full_text,
        "links": sorted(set(links)),
        "directory": directory,
    }


def extract_all(
    input_file=CIPD_PEOPLE_LINKS_FILE,
    output_file=CIPD_PEOPLE_DATA_FILE,
    checkpoint_file=CHECKPOINT_FILE,
    limit=None,
    dry_run=False,
):
    input_data = _load_json(input_file, {"articles": []})
    urls = input_data.get("articles", [])
    if limit is not None:
        urls = urls[:limit]

    checkpoint = _load_json(checkpoint_file, {})
    processed = dict(checkpoint)

    for idx, url in enumerate(urls, start=1):
        if url in processed:
            continue
        try:
            processed[url] = extract_article(url, dry_run=dry_run)
        except Exception as exc:
            processed[url] = {
                "url": url,
                "title": "",
                "author": "",
                "date": "",
                "year": None,
                "category": "",
                "tags": [],
                "description": "",
                "summary": "",
                "full_text": "",
                "links": [],
                "directory": "",
                "error": str(exc),
            }
        _save_json(checkpoint_file, processed)
        if idx % 25 == 0:
            time.sleep(0.2)

    articles = [processed[u] for u in urls if u in processed]
    payload = {"total_articles": len(articles), "articles": articles}
    _save_json(output_file, payload)
    return articles


def filter_to_year(articles, year):
    return [a for a in articles if a.get("year") == year]


def main(limit=None):
    articles = extract_all(limit=limit)
    print(f"Saved {len(articles)} peoplemanagement articles -> {CIPD_PEOPLE_DATA_FILE}")


if __name__ == "__main__":
    main()
