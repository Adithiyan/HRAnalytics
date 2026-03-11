"""
cipd/community_extractor.py - Extract CIPD community blog content with resume.
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

from config import CIPD_COMMUNITY_DATA_FILE, CIPD_COMMUNITY_LINKS_FILE, PROJECT_ROOT

CHECKPOINT_FILE = os.path.join(PROJECT_ROOT, "src", "checkpoints", "cipd_community_articles.json")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
BASE_URL = "https://community.cipd.co.uk"


def _load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _parse_date_and_year(soup):
    # Preferred: <time datetime="...">
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        raw = time_tag.get("datetime", "").strip()
        text = time_tag.get_text(" ", strip=True)
        year = None
        if len(raw) >= 4 and raw[:4].isdigit():
            year = int(raw[:4])
        return text, year

    # Fallback: "22 Jun 2021" in content-date style fields
    date_candidates = []
    for selector in ["div.content-date", "p:has(.author)", "p"]:
        for node in soup.select(selector):
            txt = node.get_text(" ", strip=True)
            if txt:
                date_candidates.append(txt)

    pattern = re.compile(r"(\d{1,2}\s+[A-Za-z]{3}\s*,?\s*\d{4})")
    for candidate in date_candidates:
        m = pattern.search(candidate)
        if not m:
            continue
        text = m.group(1).replace(" ,", ",").strip()
        for fmt in ("%d %b %Y", "%d %b, %Y"):
            try:
                parsed = dt.datetime.strptime(text, fmt)
                return parsed.strftime("%d %b %Y"), parsed.year
            except ValueError:
                pass
    return "", None


def _extract_author(soup):
    # Common community pattern: "By NAME 12 Mar, 2025"
    candidate = soup.select_one("p:has(.author)")
    if candidate:
        text = candidate.get_text(" ", strip=True)
        m = re.search(r"By\s+(.+?)\s+\d{1,2}\s+[A-Za-z]{3},?\s+\d{4}", text)
        if m:
            return m.group(1).strip()
    for sel in [".author-name", ".author", "meta[name='author']"]:
        el = soup.select_one(sel)
        if not el:
            continue
        if el.name == "meta":
            val = el.get("content", "").strip()
        else:
            val = el.get_text(" ", strip=True)
        if val:
            return val
    return ""


def extract_article(url, dry_run=False):
    if dry_run:
        return {
            "url": url,
            "title": "Mock Community Article",
            "author": "Mock Author",
            "date": "01 Jan 2025",
            "year": 2025,
            "category": "",
            "tags": [],
            "description": "Mock description",
            "summary": "Mock summary",
            "full_text": "Mock content",
            "links": [],
            "directory": "CIPD Community",
        }

    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title = ""
    for sel in ["h3.name", "h1", "title"]:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            title = el.get_text(" ", strip=True)
            break

    date_text, year = _parse_date_and_year(soup)
    author = _extract_author(soup)
    tags = [t.get_text(" ", strip=True) for t in soup.select("ul.tag-list li") if t.get_text(strip=True)]

    content_root = (
        soup.select_one("div.content > p")
        or soup.select_one("div.content")
        or soup.select_one("article")
        or soup.body
    )
    paragraphs = []
    if content_root:
        for p in content_root.find_all("p"):
            txt = p.get_text(" ", strip=True)
            if txt:
                paragraphs.append(unidecode(txt))
    full_text = "\n\n".join(paragraphs).strip()
    summary = (full_text[:300] + "...") if len(full_text) > 300 else full_text

    links = []
    if content_root:
        for a in content_root.find_all("a", href=True):
            href = a.get("href", "").strip()
            if not href or href.startswith("#"):
                continue
            links.append(urljoin(BASE_URL, href))

    breadcrumb = [x.get_text(" ", strip=True) for x in soup.select("div.row ul li") if x.get_text(strip=True)]
    directory = " > ".join([x for x in breadcrumb if x.lower() != "home"])

    return {
        "url": url,
        "title": title,
        "author": author,
        "date": date_text,
        "year": year,
        "category": "",
        "tags": tags,
        "description": summary,
        "summary": summary,
        "full_text": full_text,
        "links": sorted(set(links)),
        "directory": directory,
    }


def extract_all(
    input_file=CIPD_COMMUNITY_LINKS_FILE,
    output_file=CIPD_COMMUNITY_DATA_FILE,
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
    print(f"Saved {len(articles)} community articles -> {CIPD_COMMUNITY_DATA_FILE}")


if __name__ == "__main__":
    main()
