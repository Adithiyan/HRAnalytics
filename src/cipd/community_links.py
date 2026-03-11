"""
cipd/community_links.py - Collect CIPD community blog article URLs with resume.

This stage is intentionally independent from legacy root-level JSON files so the
entire community pipeline can be run from src/ as part of the 2025 orchestrator.
"""

import json
import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import CIPD_COMMUNITY_LINKS_FILE, PROJECT_ROOT

BASE_URL = "https://community.cipd.co.uk"
BLOG_URL_TEMPLATE = "https://community.cipd.co.uk/cipd-blogs?pifragment-36={page}"
CHECKPOINT_FILE = os.path.join(PROJECT_ROOT, "src", "checkpoints", "cipd_community_links.json")

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
    return {"last_page": 0, "links": []}


def _save_checkpoint(path, last_page, links):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"last_page": int(last_page), "links": sorted(set(links))}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _extract_links(html):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("div.col-md-15.blog-post h4 a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        links.append(urljoin(BASE_URL, href))
    return links


def _discover_total_pages(html):
    soup = BeautifulSoup(html, "html.parser")
    page_numbers = set()
    for a in soup.select("a[href*='pifragment-36=']"):
        href = a.get("href", "")
        match = re.search(r"pifragment-36=(\d+)", href)
        if match:
            page_numbers.add(int(match.group(1)))
    return max(page_numbers) if page_numbers else 1


def collect_links(
    max_pages=None,
    output_file=CIPD_COMMUNITY_LINKS_FILE,
    checkpoint_file=CHECKPOINT_FILE,
    dry_run=False,
):
    """
    Collect CIPD community blog post URLs.

    Args:
        max_pages (int|None): Optional upper bound for crawl depth.
        output_file (str): JSON destination.
        checkpoint_file (str): Resume state destination.
        dry_run (bool): If True, produce deterministic fake rows without network.
    """
    cp = _load_checkpoint(checkpoint_file)
    links = list(cp.get("links", []))
    start_page = int(cp.get("last_page", 0)) + 1

    if dry_run:
        total = max_pages or 2
        for page in range(start_page, total + 1):
            links.append(f"{BASE_URL}/cipd-blogs/b/test/posts/mock-{page}")
            _save_checkpoint(checkpoint_file, page, links)
        _write_output(output_file, links)
        return sorted(set(links))

    session = requests.Session()
    session.headers.update(HEADERS)

    # Prime first page for total page discovery unless resuming mid-run.
    discovered_total = 1
    if start_page == 1:
        resp = session.get(BLOG_URL_TEMPLATE.format(page=1), timeout=20)
        resp.raise_for_status()
        links.extend(_extract_links(resp.text))
        discovered_total = _discover_total_pages(resp.text)
        _save_checkpoint(checkpoint_file, 1, links)
        start_page = 2
    else:
        # If resuming, only rely on provided max_pages; otherwise crawl until no links.
        discovered_total = max_pages or 10_000

    end_page = discovered_total
    if max_pages is not None:
        end_page = min(end_page, max_pages)

    for page in range(start_page, end_page + 1):
        url = BLOG_URL_TEMPLATE.format(page=page)
        resp = session.get(url, timeout=20)
        if resp.status_code >= 400:
            break
        page_links = _extract_links(resp.text)
        if not page_links:
            break
        links.extend(page_links)
        _save_checkpoint(checkpoint_file, page, links)
        time.sleep(0.2)

    _write_output(output_file, links)
    return sorted(set(links))


def _write_output(output_file, links):
    unique_links = sorted(set(links))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    payload = {"total_articles": len(unique_links), "articles": unique_links}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(max_pages=None):
    urls = collect_links(max_pages=max_pages)
    print(f"Saved {len(urls)} community blog links -> {CIPD_COMMUNITY_LINKS_FILE}")


if __name__ == "__main__":
    main()
