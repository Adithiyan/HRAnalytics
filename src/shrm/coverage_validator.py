"""
shrm/coverage_validator.py - Validate SHRM legacy-to-src parity contract.
"""

import os

from config import (
    SHRM_LEGACY_SECTION_STAGE_MAP,
    SHRM_REQUIRED_ARTICLE_SELECTORS,
    SHRM_REQUIRED_LINK_SELECTORS,
    SHRM_REQUIRED_SECTION_CONTRACT,
    SHRM_TOPICS_URL,
)


def evaluate_status(crawled_count, year_count):
    if crawled_count <= 0:
        return "missing"
    if year_count <= 0:
        return "covered_zero_2025"
    return "covered"


def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _contains_all(text, required_tokens):
    missing = [token for token in required_tokens if token not in text]
    return (len(missing) == 0), missing


def build_static_parity_evidence(
    legacy_link_file,
    legacy_article_file,
    src_link_file,
    src_article_file,
):
    """
    Verify that src SHRM modules preserve legacy page surfaces/selectors.
    """
    legacy_link_text = _read_text(legacy_link_file)
    legacy_article_text = _read_text(legacy_article_file)
    src_link_text = _read_text(src_link_file)
    src_article_text = _read_text(src_article_file)

    # All required selectors should still be present in src files.
    link_ok, link_missing = _contains_all(src_link_text, SHRM_REQUIRED_LINK_SELECTORS)
    article_ok, article_missing = _contains_all(src_article_text, SHRM_REQUIRED_ARTICLE_SELECTORS)

    # Legacy files should also include those surfaces (safety check for drift).
    legacy_link_ok, legacy_link_missing = _contains_all(legacy_link_text, SHRM_REQUIRED_LINK_SELECTORS)
    legacy_article_ok, legacy_article_missing = _contains_all(
        legacy_article_text, SHRM_REQUIRED_ARTICLE_SELECTORS
    )

    src_url_reference_ok = (SHRM_TOPICS_URL in src_link_text) or ("SHRM_TOPICS_URL" in src_link_text)
    start_url_ok = (SHRM_TOPICS_URL in legacy_link_text) and src_url_reference_ok

    return {
        "start_url": SHRM_TOPICS_URL,
        "start_url_matches_legacy_and_src": start_url_ok,
        "link_selector_parity_ok": link_ok and legacy_link_ok,
        "article_selector_parity_ok": article_ok and legacy_article_ok,
        "missing_in_src_link_collector": link_missing,
        "missing_in_src_article_extractor": article_missing,
        "missing_in_legacy_link_script": legacy_link_missing,
        "missing_in_legacy_article_script": legacy_article_missing,
    }


def build_coverage_report(
    target_year,
    links_count,
    articles_2025_count,
    static_evidence,
):
    """
    Produce SHRM contract coverage report aligned with CIPD status semantics.
    """
    topics_status = "covered" if static_evidence.get("link_selector_parity_ok") else "missing"
    article_surface_status = "covered" if static_evidence.get("article_selector_parity_ok") else "missing"
    year_status = evaluate_status(links_count, articles_2025_count)

    rows = [
        {
            "section": "topics-tools-search",
            "stage": SHRM_LEGACY_SECTION_STAGE_MAP["topics-tools-search"],
            "status": topics_status,
            "crawled_count": links_count,
            "year_count": articles_2025_count,
        },
        {
            "section": "article-pages",
            "stage": SHRM_LEGACY_SECTION_STAGE_MAP["article-pages"],
            "status": article_surface_status,
            "crawled_count": links_count,
            "year_count": articles_2025_count,
        },
        {
            "section": "year-coverage",
            "stage": "shrm-2025-filter",
            "status": year_status,
            "crawled_count": links_count,
            "year_count": articles_2025_count,
        },
    ]

    summary = {"covered": 0, "covered_zero_2025": 0, "missing": 0}
    for row in rows:
        summary[row["status"]] += 1

    return {
        "target_year": target_year,
        "required_sections": SHRM_REQUIRED_SECTION_CONTRACT,
        "sections": rows,
        "summary": summary,
        "static_parity_evidence": static_evidence,
    }


def resolve_default_parity_paths(project_root):
    return {
        "legacy_link_file": os.path.join(project_root, "legacy", "scripts", "shrm1.py"),
        "legacy_article_file": os.path.join(project_root, "legacy", "scripts", "shrm2optimized.py"),
        "src_link_file": os.path.join(project_root, "src", "shrm", "link_collector.py"),
        "src_article_file": os.path.join(project_root, "src", "shrm", "article_extractor.py"),
    }
