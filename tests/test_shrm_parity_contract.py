import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from config import (  # noqa: E402
    SHRM_LEGACY_SECTION_STAGE_MAP,
    SHRM_REQUIRED_SECTION_CONTRACT,
    SHRM_TOPICS_URL,
)
from shrm.coverage_validator import (  # noqa: E402
    build_static_parity_evidence,
    resolve_default_parity_paths,
)


def test_shrm_section_contract_mapping_complete():
    legacy_sections = {"topics-tools-search", "article-pages"}
    assert legacy_sections.issubset(set(SHRM_REQUIRED_SECTION_CONTRACT))
    assert legacy_sections.issubset(set(SHRM_LEGACY_SECTION_STAGE_MAP.keys()))


def test_shrm_static_selector_parity_against_legacy_and_src():
    project_root = os.path.abspath(".")
    paths = resolve_default_parity_paths(project_root)
    evidence = build_static_parity_evidence(**paths)

    assert evidence["start_url"] == SHRM_TOPICS_URL
    assert evidence["start_url_matches_legacy_and_src"] is True
    assert evidence["link_selector_parity_ok"] is True
    assert evidence["article_selector_parity_ok"] is True
