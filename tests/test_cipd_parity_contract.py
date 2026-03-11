import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from config import CIPD_LEGACY_SECTION_STAGE_MAP, CIPD_REQUIRED_SECTION_CONTRACT  # noqa: E402


def test_legacy_sections_mapped_to_new_stages():
    # Legacy surfaces covered across cipd.py + cipd/*.py
    legacy_sections = {
        "factsheets",
        "guides",
        "reports",
        "tools",
        "podcasts",
        "webinars",
        "case-studies",
        "evidence-reviews",
        "bitesize-research",
        "employment-law",
        "community-blog",
        "people-management",
        "podcast-episodes",
    }
    assert legacy_sections.issubset(set(CIPD_REQUIRED_SECTION_CONTRACT))
    assert legacy_sections.issubset(set(CIPD_LEGACY_SECTION_STAGE_MAP.keys()))


def test_stage_mapping_is_complete_for_contract():
    for section in CIPD_REQUIRED_SECTION_CONTRACT:
        assert section in CIPD_LEGACY_SECTION_STAGE_MAP
        assert CIPD_LEGACY_SECTION_STAGE_MAP[section]
