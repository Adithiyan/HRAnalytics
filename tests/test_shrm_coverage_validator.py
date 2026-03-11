import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from shrm.coverage_validator import build_coverage_report, evaluate_status  # noqa: E402


def test_shrm_evaluate_status_logic():
    assert evaluate_status(0, 0) == "missing"
    assert evaluate_status(20, 0) == "covered_zero_2025"
    assert evaluate_status(20, 5) == "covered"


def test_shrm_coverage_report_structure():
    static_evidence = {
        "start_url": "https://www.shrm.org/topics-tools",
        "start_url_matches_legacy_and_src": True,
        "link_selector_parity_ok": True,
        "article_selector_parity_ok": True,
        "missing_in_src_link_collector": [],
        "missing_in_src_article_extractor": [],
        "missing_in_legacy_link_script": [],
        "missing_in_legacy_article_script": [],
    }
    report = build_coverage_report(
        target_year=2025,
        links_count=100,
        articles_2025_count=25,
        static_evidence=static_evidence,
    )
    assert report["summary"]["missing"] == 0
    assert len(report["sections"]) == 3
