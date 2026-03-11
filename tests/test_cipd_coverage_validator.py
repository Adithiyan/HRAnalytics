import json
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath("src"))

from cipd.coverage_validator import build_coverage_report, evaluate_status  # noqa: E402


def test_evaluate_status_logic():
    assert evaluate_status(0, 0) == "missing"
    assert evaluate_status(5, 0) == "covered_zero_2025"
    assert evaluate_status(5, 2) == "covered"


def test_build_coverage_report_marks_sections():
    base = os.path.join(os.path.abspath("."), "tests_tmp", f"coverage_{uuid.uuid4().hex}")
    os.makedirs(base, exist_ok=True)
    links_file = os.path.join(base, "links.json")
    links_payload = {
        "https://www.cipd.org/uk/knowledge/factsheets/": {"articles": ["https://www.cipd.org/uk/knowledge/factsheets/a"]},
        "https://www.cipd.org/uk/knowledge/guides/": {"articles": ["https://www.cipd.org/uk/knowledge/guides/a"]},
        "https://www.cipd.org/uk/knowledge/reports/": {"articles": ["https://www.cipd.org/uk/knowledge/reports/a"]},
        "https://www.cipd.org/uk/knowledge/tools/": {"articles": ["https://www.cipd.org/uk/knowledge/tools/a"]},
        "https://www.cipd.org/uk/knowledge/podcasts/": {"articles": ["https://www.cipd.org/uk/knowledge/podcasts/a"]},
        "https://www.cipd.org/uk/knowledge/webinars/": {"articles": ["https://www.cipd.org/uk/knowledge/webinars/a"]},
        "https://www.cipd.org/uk/knowledge/case-studies/": {"articles": ["https://www.cipd.org/uk/knowledge/case-studies/a"]},
        "https://www.cipd.org/uk/knowledge/evidence-reviews/": {"articles": ["https://www.cipd.org/uk/knowledge/evidence-reviews/a"]},
        "https://www.cipd.org/uk/knowledge/bitesize-research/": {"articles": ["https://www.cipd.org/uk/knowledge/bitesize-research/a"]},
        "https://www.cipd.org/uk/knowledge/employment-law/": {"articles": ["https://www.cipd.org/uk/knowledge/employment-law/a"]},
    }
    with open(links_file, "w", encoding="utf-8") as f:
        json.dump(links_payload, f)

    knowledge_articles = [
        {"url": "https://www.cipd.org/uk/knowledge/factsheets/a", "year": 2025},
        {"url": "https://www.cipd.org/uk/knowledge/guides/a", "year": 2025},
        {"url": "https://www.cipd.org/uk/knowledge/webinars/a", "year": 2024},
    ]
    community_articles = [{"url": "x", "year": 2025}]
    people_articles = [{"url": "y", "year": 2024}]
    podcast_payload = {"series": [{"serie_year": 2025, "serie_podcast": [{"podcast_title": "e1"}]}]}

    report = build_coverage_report(
        target_year=2025,
        knowledge_links_file=links_file,
        knowledge_articles=knowledge_articles,
        community_articles=community_articles,
        people_articles=people_articles,
        podcast_payload=podcast_payload,
        include_podcast_stage=True,
    )

    by_section = {row["section"]: row for row in report["sections"]}
    assert by_section["factsheets"]["status"] == "covered"
    assert by_section["webinars"]["status"] == "covered_zero_2025"
    assert by_section["community-blog"]["status"] == "covered"
    assert by_section["people-management"]["status"] == "covered_zero_2025"
    assert by_section["podcast-episodes"]["status"] == "covered"
