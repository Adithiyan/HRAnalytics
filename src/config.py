"""
config.py — Shared configuration for all scrapers.

All file paths are computed as absolute paths relative to the project root so
that every script in src/ works correctly regardless of the working directory
from which it is invoked.

Credentials: CIPD credentials are stored here directly (same as the originals).
             SHRM credentials are read from misc/.env with fallback defaults.
"""

import os

# ---------------------------------------------------------------------------
# Project root — parent of this src/ directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# CIPD credentials & URLs
# ---------------------------------------------------------------------------
CIPD_USERNAME    = "Jade.yy423@gmail.com"
CIPD_PASSWORD    = "jadeCIPD123"

CIPD_LOGIN_URL     = "https://www.cipd.org/login"
CIPD_HOME_URL      = "https://www.cipd.org/uk"
CIPD_KNOWLEDGE_URL = "https://www.cipd.org/uk/policy-and-insights/"

# ---------------------------------------------------------------------------
# SHRM credentials & URLs  (sourced from misc/.env)
# ---------------------------------------------------------------------------
def _load_env():
    env_path = os.path.join(PROJECT_ROOT, "misc", ".env")
    env = {}
    if os.path.exists(env_path):
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    env[key.strip()] = val.strip()
    return env

_env = _load_env()
SHRM_USERNAME = _env.get("USERNAME", "Yaoyao.irhr@gmail.com")
SHRM_PASSWORD = _env.get("PASS", "SHRM123yyy")

SHRM_TOPICS_URL = "https://www.shrm.org/topics-tools"

# SHRM parity contract (legacy/scripts/shrm1.py/shrm2.py/shrm2optimized.py -> src/shrm/*)
SHRM_REQUIRED_SECTION_CONTRACT = [
    "topics-tools-search",
    "article-pages",
]

SHRM_LEGACY_SECTION_STAGE_MAP = {
    "topics-tools-search": "shrm-link-collector",
    "article-pages": "shrm-article-extractor",
}

SHRM_REQUIRED_LINK_SELECTORS = [
    "atomic-result",
    "atomic-pager",
    'button[aria-label="Next"]',
]

SHRM_REQUIRED_ARTICLE_SELECTORS = [
    "h1.content__title",
    "ol.cmp-breadcrumb__list li.cmp-breadcrumb__item span[itemprop=\"name\"]",
    "span.content__date",
    "span.content__author",
    "div.pretitle a[data-contentfiltertag]",
    "a[aria-label=\"button tag\"]",
    "div.cmp-text",
]

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SHRM_OUTPUT_COVERAGE_PATTERN = os.path.join(OUTPUT_DIR, "shrm_coverage_{timestamp}.json")

# CIPD — authentication & intermediate files kept alongside original scripts
CIPD_COOKIES_FILE       = os.path.join(PROJECT_ROOT, "cipd", "cookies.pkl")
CIPD_STORAGE_STATE      = os.path.join(PROJECT_ROOT, "storage_state.json")
CIPD_MAIN_LINKS_FILE    = os.path.join(PROJECT_ROOT, "cipd", "main_links_views_uk.txt")
CIPD_ARTICLE_LINKS_FILE = os.path.join(PROJECT_ROOT, "cipd", "scraped_links_views_uk.json")
CIPD_ARTICLE_DATA_FILE  = os.path.join(PROJECT_ROOT, "cipd", "cipd_article_data_views_uk.json")

CIPD_BLOG_INPUT_FILE  = os.path.join(PROJECT_ROOT, "scraped_links_cipd_community_target.json")
CIPD_BLOG_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "cipd_articles_static_scrape.json")

# CIPD parity additions (community, peoplemanagement, podcasts, coverage)
CIPD_COMMUNITY_LINKS_FILE   = os.path.join(PROJECT_ROOT, "cipd", "scraped_links_cipd_community.json")
CIPD_COMMUNITY_DATA_FILE    = os.path.join(PROJECT_ROOT, "cipd", "cipd_article_data_community.json")
CIPD_PEOPLE_LINKS_FILE      = os.path.join(PROJECT_ROOT, "cipd", "scraped_links_people_management.json")
CIPD_PEOPLE_DATA_FILE       = os.path.join(PROJECT_ROOT, "cipd", "cipd_article_data_people_management.json")
CIPD_PODCAST_EPISODES_FILE  = os.path.join(PROJECT_ROOT, "cipd", "cipd_podcast_episode_data.json")

CIPD_OUTPUT_KNOWLEDGE_2025_PATTERN = os.path.join(OUTPUT_DIR, "cipd_2025_{timestamp}.json")
CIPD_OUTPUT_COMMUNITY_2025_PATTERN = os.path.join(OUTPUT_DIR, "cipd_community_2025_{timestamp}.json")
CIPD_OUTPUT_PEOPLE_2025_PATTERN    = os.path.join(OUTPUT_DIR, "cipd_people_management_2025_{timestamp}.json")
CIPD_OUTPUT_PODCAST_2025_PATTERN   = os.path.join(OUTPUT_DIR, "cipd_podcast_episodes_2025_{timestamp}.json")
CIPD_OUTPUT_COVERAGE_PATTERN       = os.path.join(OUTPUT_DIR, "cipd_coverage_{timestamp}.json")

# Required legacy coverage contract:
# - 10 knowledge sections from legacy views/knowledge scraping
# - 1 CIPD community blog surface
# - 1 People Management surface
# - 1 podcast episode-level surface from legacy cipd.py
CIPD_REQUIRED_KNOWLEDGE_SECTIONS = [
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
]

CIPD_REQUIRED_EXTRA_SECTIONS = [
    "community-blog",
    "people-management",
    "podcast-episodes",
]

CIPD_REQUIRED_SECTION_CONTRACT = CIPD_REQUIRED_KNOWLEDGE_SECTIONS + CIPD_REQUIRED_EXTRA_SECTIONS

# Legacy-to-new stage mapping contract used by parity tests and coverage reports.
CIPD_LEGACY_SECTION_STAGE_MAP = {
    "factsheets": "knowledge",
    "guides": "knowledge",
    "reports": "knowledge",
    "tools": "knowledge",
    "podcasts": "knowledge",
    "webinars": "knowledge",
    "case-studies": "knowledge",
    "evidence-reviews": "knowledge",
    "bitesize-research": "knowledge",
    "employment-law": "knowledge",
    "community-blog": "community",
    "people-management": "people-management",
    "podcast-episodes": "podcast-episodes",
}

