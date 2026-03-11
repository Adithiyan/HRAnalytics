"""
cipd/podcast_episodes.py - Episode-level CIPD podcast scraper for parity with cipd.py.

Output keeps legacy-compatible keys:
  - series metadata: serie_title, serie_link, serie_date, serie_tags, ...
  - episode fields: podcast_title, podcast_audio_link, podcast_duration,
                    podcast_description, podcast_transcript
"""

import datetime as dt
import json
import os
import re
import time
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from config import CIPD_PODCAST_EPISODES_FILE, PROJECT_ROOT

ROOT_URL = "https://www.cipd.org"
PODCAST_LIST_TEMPLATE = "https://www.cipd.org/uk/knowledge/podcasts/?page={page}"
CHECKPOINT_FILE = os.path.join(PROJECT_ROOT, "src", "checkpoints", "cipd_podcast_episodes.json")


def _load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _parse_year(date_text):
    if not date_text:
        return None
    for fmt in ("%d %b %Y", "%d %b, %Y", "%d %B %Y", "%B %d, %Y"):
        try:
            return dt.datetime.strptime(date_text.strip(), fmt).year
        except ValueError:
            pass
    m = re.search(r"(20\d{2}|19\d{2})", date_text)
    return int(m.group(1)) if m else None


def _parse_series_cards(html):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    root = soup.find("div", class_="listing__results")
    if not root:
        return out

    for card in root.find_all("div", class_="card card--full"):
        head = card.find("div", class_="card__content-head")
        anchor = head.find("a") if head else None
        href = anchor.get("href", "").strip() if anchor else ""
        if not href:
            continue
        date_text = ""
        date_div = card.find("div", class_="card__date")
        if date_div:
            date_text = date_div.get_text(" ", strip=True)
        desc = ""
        desc_div = card.find("div", class_="card__desc")
        if desc_div:
            desc = desc_div.get_text(" ", strip=True)

        tags = [x.get_text(" ", strip=True) for x in card.find_all("div", class_="card__tag") if x.get_text(strip=True)]

        out.append(
            {
                "serie_title": anchor.get("title", "").strip() or anchor.get_text(" ", strip=True),
                "serie_link": urljoin(ROOT_URL, href),
                "serie_date": date_text,
                "serie_year": _parse_year(date_text),
                "serie_description": desc,
                "serie_tags": tags,
            }
        )
    return out


def _extract_duration_and_transcript(section):
    duration = ""
    transcript = ""

    accordion_wrapper = section.find("div", class_="accordion__wrapper")
    if accordion_wrapper:
        p = accordion_wrapper.find("p")
        if p:
            txt = p.get_text(" ", strip=True)
            if txt.startswith("Duration"):
                duration = txt

    content_wrapper = section.find("div", class_="accordion__content-wrapper")
    if content_wrapper:
        paragraphs = [p.get_text(" ", strip=True) for p in content_wrapper.find_all("p") if p.get_text(strip=True)]
        transcript = "\n".join(paragraphs).strip()

    return duration, transcript


def _parse_series_page(html):
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main", id="main")
    if not main:
        return []

    sections = main.find_all("section")
    audio_sections = [s for s in sections if s.find("div", class_="audio")]

    if not audio_sections:
        return []

    # Single-episode layout: one audio section + page intro description.
    if len(audio_sections) == 1:
        section = audio_sections[0]
        iframe = section.find("iframe")
        duration, transcript = _extract_duration_and_transcript(section)

        description = ""
        intro = main.find("div", class_="page-intro__content-wrapper")
        if intro:
            description = "\n".join([p.get_text(" ", strip=True) for p in intro.find_all("p") if p.get_text(strip=True)]).strip()

        title = ""
        head = soup.find("head")
        if head and head.find("title"):
            title = head.find("title").get_text(" ", strip=True)

        return [
            {
                "podcast_title": title,
                "podcast_audio_link": iframe.get("src", "").strip() if iframe else "",
                "podcast_duration": duration,
                "podcast_description": description,
                "podcast_transcript": transcript,
            }
        ]

    # Multi-episode layout: pair audio sections with nearby rich text descriptions.
    episodes = []
    descriptions = []
    for section in sections:
        if section.find("div", class_="audio"):
            iframe = section.find("iframe")
            duration, transcript = _extract_duration_and_transcript(section)
            title_node = section.find("h2", class_="audio__title")
            episodes.append(
                {
                    "podcast_title": title_node.get_text(" ", strip=True) if title_node else "",
                    "podcast_audio_link": iframe.get("src", "").strip() if iframe else "",
                    "podcast_duration": duration,
                    "podcast_description": "",
                    "podcast_transcript": transcript,
                }
            )
        elif section.find("div", class_="rich-text__wrapper"):
            p_list = [p.get_text(" ", strip=True) for p in section.find_all("p") if p.get_text(strip=True)]
            descriptions.append("\n".join(p_list).strip())

    if len(descriptions) == len(episodes):
        for idx, desc in enumerate(descriptions):
            episodes[idx]["podcast_description"] = desc

    return episodes


def scrape_podcast_episodes(
    output_file=CIPD_PODCAST_EPISODES_FILE,
    checkpoint_file=CHECKPOINT_FILE,
    max_pages=50,
    dry_run=False,
):
    """
    Crawl podcast listing pages, then parse each series page for episode-level data.
    """
    checkpoint = _load_json(checkpoint_file, {})
    processed = dict(checkpoint)

    if dry_run:
        sample_link = "https://www.cipd.org/uk/knowledge/podcasts/mock/"
        processed[sample_link] = {
            "serie_title": "Mock Podcast Series",
            "serie_link": sample_link,
            "serie_date": "01 Jan 2025",
            "serie_year": 2025,
            "serie_description": "Mock series description",
            "serie_tags": ["Podcast"],
            "serie_podcast": [
                {
                    "podcast_title": "Mock Episode",
                    "podcast_audio_link": "https://example.com/audio",
                    "podcast_duration": "Duration: 10 minutes",
                    "podcast_description": "Mock description",
                    "podcast_transcript": "Mock transcript",
                }
            ],
        }
        _save_json(checkpoint_file, processed)
        return _write_output(output_file, list(processed.values()))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        try:
            series_meta = {}
            seen_links = set()

            for page_num in range(1, max_pages + 1):
                page.goto(PODCAST_LIST_TEMPLATE.format(page=page_num), timeout=25000)
                page.wait_for_load_state("networkidle")
                html = page.content()
                cards = _parse_series_cards(html)
                if not cards:
                    break

                new_links = 0
                for item in cards:
                    if item["serie_link"] not in seen_links:
                        seen_links.add(item["serie_link"])
                        series_meta[item["serie_link"]] = item
                        new_links += 1
                # Stop if pagination starts repeating.
                if new_links == 0:
                    break
                time.sleep(0.2)

            for idx, (series_link, meta) in enumerate(series_meta.items(), start=1):
                if series_link in processed:
                    continue
                page.goto(series_link, timeout=25000)
                page.wait_for_load_state("networkidle")
                episodes = _parse_series_page(page.content())
                meta["serie_podcast"] = episodes
                processed[series_link] = meta
                _save_json(checkpoint_file, processed)
                if idx % 20 == 0:
                    time.sleep(0.2)

        finally:
            browser.close()

    return _write_output(output_file, list(processed.values()))


def _write_output(path, series_list):
    series_list = sorted(series_list, key=lambda x: (x.get("serie_year") or 0), reverse=True)
    total_episodes = sum(len(x.get("serie_podcast", [])) for x in series_list)
    payload = {
        "total_series": len(series_list),
        "total_episodes": total_episodes,
        "series": series_list,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def filter_to_year(series_list, year):
    filtered = []
    for series in series_list:
        if series.get("serie_year") == year:
            filtered.append(series)
            continue
        # Fallback: infer from free-text date if serie_year is missing.
        if series.get("serie_year") is None and _parse_year(series.get("serie_date", "")) == year:
            filtered.append(series)
    return filtered


def main(max_pages=50):
    payload = scrape_podcast_episodes(max_pages=max_pages)
    print(
        "Saved "
        f"{payload.get('total_series', 0)} podcast series / "
        f"{payload.get('total_episodes', 0)} episodes -> {CIPD_PODCAST_EPISODES_FILE}"
    )


if __name__ == "__main__":
    main()
