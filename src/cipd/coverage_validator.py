"""
cipd/coverage_validator.py - Validate CIPD section coverage against legacy contract.
"""

import json
import os
from collections import Counter
from urllib.parse import urlparse

from config import (
    CIPD_REQUIRED_KNOWLEDGE_SECTIONS,
    CIPD_REQUIRED_EXTRA_SECTIONS,
    CIPD_REQUIRED_SECTION_CONTRACT,
    CIPD_LEGACY_SECTION_STAGE_MAP,
)


def knowledge_section_from_url(url):
    path = urlparse(url).path.strip("/")
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "uk" and parts[1] == "knowledge":
        return parts[2]
    return ""


def count_knowledge_sections(article_links_json):
    data = _load_json(article_links_json, {})
    urls = []
    for value in data.values():
        if isinstance(value, dict):
            urls.extend(value.get("articles", []))
    counter = Counter()
    for url in urls:
        section = knowledge_section_from_url(url)
        if section:
            counter[section] += 1
    return counter


def count_years_by_section(articles, section_key_func):
    out = {}
    for row in articles:
        section = section_key_func(row)
        if not section:
            continue
        bucket = out.setdefault(section, {"crawled": 0, "year": 0})
        bucket["crawled"] += 1
        if row.get("year") is not None:
            bucket.setdefault("years", Counter())
            bucket["years"][row.get("year")] += 1
    return out


def evaluate_status(crawled_count, year_count):
    if crawled_count <= 0:
        return "missing"
    if year_count <= 0:
        return "covered_zero_2025"
    return "covered"


def build_coverage_report(
    target_year,
    knowledge_links_file,
    knowledge_articles,
    community_articles,
    people_articles,
    podcast_payload,
    include_podcast_stage=True,
):
    """
    Build a contract-level coverage report.
    """
    required = list(CIPD_REQUIRED_SECTION_CONTRACT)
    report_rows = []

    # Knowledge sections coverage.
    link_counts = count_knowledge_sections(knowledge_links_file)
    year_counts = Counter()
    for article in knowledge_articles:
        section = knowledge_section_from_url(article.get("url", ""))
        if section and article.get("year") == target_year:
            year_counts[section] += 1

    for section in CIPD_REQUIRED_KNOWLEDGE_SECTIONS:
        crawled = int(link_counts.get(section, 0))
        year_value = int(year_counts.get(section, 0))
        report_rows.append(
            {
                "section": section,
                "stage": CIPD_LEGACY_SECTION_STAGE_MAP[section],
                "status": evaluate_status(crawled, year_value),
                "crawled_count": crawled,
                "year_count": year_value,
            }
        )

    # Community coverage.
    community_crawled = len(community_articles)
    community_year = sum(1 for a in community_articles if a.get("year") == target_year)
    report_rows.append(
        {
            "section": "community-blog",
            "stage": CIPD_LEGACY_SECTION_STAGE_MAP["community-blog"],
            "status": evaluate_status(community_crawled, community_year),
            "crawled_count": community_crawled,
            "year_count": community_year,
        }
    )

    # People Management coverage.
    people_crawled = len(people_articles)
    people_year = sum(1 for a in people_articles if a.get("year") == target_year)
    report_rows.append(
        {
            "section": "people-management",
            "stage": CIPD_LEGACY_SECTION_STAGE_MAP["people-management"],
            "status": evaluate_status(people_crawled, people_year),
            "crawled_count": people_crawled,
            "year_count": people_year,
        }
    )

    # Podcast episode-level coverage.
    if include_podcast_stage:
        series = podcast_payload.get("series", [])
        crawled = sum(len(s.get("serie_podcast", [])) for s in series)
        year_count = 0
        for s in series:
            if s.get("serie_year") == target_year:
                year_count += len(s.get("serie_podcast", []))
        status = evaluate_status(crawled, year_count)
    else:
        crawled = 0
        year_count = 0
        status = "missing"

    report_rows.append(
        {
            "section": "podcast-episodes",
            "stage": CIPD_LEGACY_SECTION_STAGE_MAP["podcast-episodes"],
            "status": status if include_podcast_stage else "missing",
            "crawled_count": crawled,
            "year_count": year_count,
            "omitted_intentionally": not include_podcast_stage,
        }
    )

    status_counter = Counter(row["status"] for row in report_rows)
    return {
        "target_year": target_year,
        "required_sections": required,
        "sections": report_rows,
        "summary": {
            "covered": status_counter.get("covered", 0),
            "covered_zero_2025": status_counter.get("covered_zero_2025", 0),
            "missing": status_counter.get("missing", 0),
        },
    }


def _load_json(path, default):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

