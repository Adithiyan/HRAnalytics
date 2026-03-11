"""
cipd/peoplemanagement_links.py - Collect People Management article URLs with resume.
"""

import json
import os
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import CIPD_PEOPLE_LINKS_FILE, PROJECT_ROOT

BASE_URL = "https://www.peoplemanagement.co.uk"
START_URL = f"{BASE_URL}/search/articles"
CHECKPOINT_FILE = os.path.join(PROJECT_ROOT, "src", "checkpoints", "cipd_people_links.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"current_url": START_URL, "links": [], "done": False}


def _save_checkpoint(path, current_url, links, done=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"current_url": current_url, "links": sorted(set(links)), "done": bool(done)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _extract_links_and_next(html):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("div.searchItem.storyContent h1 a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        links.append(urljoin(BASE_URL, href))

    next_url = None
    next_a = soup.select_one("div.paginationNext a[href]")
    if next_a:
        next_url = urljoin(BASE_URL, next_a.get("href", "").strip())
    return links, next_url


def collect_links(
    output_file=CIPD_PEOPLE_LINKS_FILE,
    checkpoint_file=CHECKPOINT_FILE,
    max_pages=None,
    dry_run=False,
):
    cp = _load_checkpoint(checkpoint_file)
    links = list(cp.get("links", []))

    if cp.get("done"):
        _write_output(output_file, links)
        return sorted(set(links))

    if dry_run:
        page_limit = max_pages or 2
        for i in range(1, page_limit + 1):
            links.append(f"{BASE_URL}/article/mock-{i}")
            _save_checkpoint(checkpoint_file, f"dry://page/{i}", links, done=(i == page_limit))
        _write_output(output_file, links)
        return sorted(set(links))

    current_url = cp.get("current_url") or START_URL
    page_count = 0
    session = requests.Session()
    session.headers.update(HEADERS)

    while current_url:
        if max_pages is not None and page_count >= max_pages:
            break
        resp = session.get(current_url, timeout=20)
        resp.raise_for_status()

        page_links, next_url = _extract_links_and_next(resp.text)
        if not page_links and not next_url:
            _save_checkpoint(checkpoint_file, current_url, links, done=True)
            break

        links.extend(page_links)
        page_count += 1
        done = not next_url
        _save_checkpoint(checkpoint_file, next_url or current_url, links, done=done)
        if done:
            break
        current_url = next_url
        time.sleep(0.2)

    _write_output(output_file, links)
    return sorted(set(links))


def _write_output(path, links):
    unique_links = sorted(set(links))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "start_url": START_URL,
        "total_articles": len(unique_links),
        "articles": unique_links,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(max_pages=None):
    urls = collect_links(max_pages=max_pages)
    print(f"Saved {len(urls)} peoplemanagement links -> {CIPD_PEOPLE_LINKS_FILE}")


if __name__ == "__main__":
    main()
