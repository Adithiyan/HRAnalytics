"""
run_cipd_2025.py - Unified CIPD 2025 orchestration with coverage reporting.

Stages:
  1) Knowledge hub (checkpointed): phase1_cipd_links.py + phase2_cipd_articles.py
  2) Community blog: cipd/community_links.py + cipd/community_extractor.py
  3) People Management: cipd/peoplemanagement_links.py + cipd/peoplemanagement_extractor.py
  4) Podcast episodes (legacy cipd.py parity): cipd/podcast_episodes.py
  5) Coverage validation: cipd/coverage_validator.py
"""

import argparse
import json
import os
from datetime import datetime

from config import (
    CIPD_ARTICLE_LINKS_FILE,
    CIPD_COMMUNITY_DATA_FILE,
    CIPD_COMMUNITY_LINKS_FILE,
    CIPD_OUTPUT_COMMUNITY_2025_PATTERN,
    CIPD_OUTPUT_COVERAGE_PATTERN,
    CIPD_OUTPUT_KNOWLEDGE_2025_PATTERN,
    CIPD_OUTPUT_PEOPLE_2025_PATTERN,
    CIPD_OUTPUT_PODCAST_2025_PATTERN,
    CIPD_PEOPLE_DATA_FILE,
    CIPD_PEOPLE_LINKS_FILE,
    CIPD_PODCAST_EPISODES_FILE,
    OUTPUT_DIR,
    PROJECT_ROOT,
)
from phase1_cipd_links import main as run_knowledge_links
from phase2_cipd_articles import main as run_knowledge_articles
from cipd.community_links import (
    CHECKPOINT_FILE as COMMUNITY_LINKS_CHECKPOINT,
    collect_links as collect_community_links,
)
from cipd.community_extractor import (
    CHECKPOINT_FILE as COMMUNITY_ARTICLES_CHECKPOINT,
    extract_all as extract_community_articles,
)
from cipd.community_extractor import filter_to_year as filter_community_to_year
from cipd.peoplemanagement_links import (
    CHECKPOINT_FILE as PEOPLE_LINKS_CHECKPOINT,
    collect_links as collect_people_links,
)
from cipd.peoplemanagement_extractor import (
    CHECKPOINT_FILE as PEOPLE_ARTICLES_CHECKPOINT,
    extract_all as extract_people_articles,
)
from cipd.peoplemanagement_extractor import filter_to_year as filter_people_to_year
from cipd.podcast_episodes import (
    CHECKPOINT_FILE as PODCAST_CHECKPOINT,
    filter_to_year as filter_podcast_to_year,
    scrape_podcast_episodes,
)
from cipd.coverage_validator import build_coverage_report


CIPD_ARTICLES_CHECKPOINT = os.path.join(PROJECT_ROOT, "src", "checkpoints", "cipd_articles.json")


def _load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_knowledge_articles_for_coverage():
    """
    Coverage should look at all processed knowledge articles (all years), not
    just filtered 2025 output.
    """
    cp = _load_json(CIPD_ARTICLES_CHECKPOINT, {})
    if isinstance(cp, dict) and cp:
        return [v for v in cp.values() if isinstance(v, dict)]

    data = _load_json(CIPD_ARTICLE_DATA_FILE, {"articles": []})
    return data.get("articles", [])


def _write_year_output(path, source, year, items, generated):
    payload = {
        "source": source,
        "year": year,
        "total_articles": len(items),
        "generated": generated,
        "articles": items,
    }
    _save_json(path, payload)
    return payload


def run_pipeline(
    target_year=2025,
    include_podcast_stage=True,
    community_max_pages=None,
    people_max_pages=None,
    limit_community_articles=None,
    limit_people_articles=None,
    podcast_max_pages=50,
    dry_run=False,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    knowledge_output = CIPD_OUTPUT_KNOWLEDGE_2025_PATTERN.format(timestamp=timestamp)
    community_output = CIPD_OUTPUT_COMMUNITY_2025_PATTERN.format(timestamp=timestamp)
    people_output = CIPD_OUTPUT_PEOPLE_2025_PATTERN.format(timestamp=timestamp)
    podcast_output = CIPD_OUTPUT_PODCAST_2025_PATTERN.format(timestamp=timestamp)
    coverage_output = CIPD_OUTPUT_COVERAGE_PATTERN.format(timestamp=timestamp)
    dry_run_dir = os.path.join(OUTPUT_DIR, "dry_run_artifacts", timestamp)

    # 1) Knowledge hub (canonical checkpointed flow)
    if dry_run:
        mock_knowledge = []
        for section in [
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
        ]:
            mock_knowledge.append(
                {
                    "url": f"https://www.cipd.org/uk/knowledge/{section}/mock",
                    "title": f"Mock {section}",
                    "date": "01 Jan, 2025",
                    "year": target_year,
                    "category": "",
                    "tags": [],
                    "description": "mock",
                    "summary": "mock",
                    "full_text": "mock",
                    "links": [],
                    "directory": "",
                }
            )
        _write_year_output(knowledge_output, "CIPD", target_year, mock_knowledge, timestamp)
        knowledge_result = {"output_file": knowledge_output}
        knowledge_articles_all = mock_knowledge
    else:
        run_knowledge_links()
        knowledge_result = run_knowledge_articles(target_year=target_year, final_output_path=knowledge_output)
        knowledge_articles_all = _load_knowledge_articles_for_coverage()

    # 2) Community blog pipeline
    community_links_file = CIPD_COMMUNITY_LINKS_FILE
    community_data_file = CIPD_COMMUNITY_DATA_FILE
    community_links_cp = None
    community_articles_cp = None
    if dry_run:
        os.makedirs(dry_run_dir, exist_ok=True)
        community_links_file = os.path.join(dry_run_dir, "scraped_links_cipd_community.json")
        community_data_file = os.path.join(dry_run_dir, "cipd_article_data_community.json")
        community_links_cp = os.path.join(dry_run_dir, "cp_cipd_community_links.json")
        community_articles_cp = os.path.join(dry_run_dir, "cp_cipd_community_articles.json")

    collect_community_links(
        max_pages=community_max_pages,
        output_file=community_links_file,
        checkpoint_file=community_links_cp or COMMUNITY_LINKS_CHECKPOINT,
        dry_run=dry_run,
    )
    community_all = extract_community_articles(
        input_file=community_links_file,
        output_file=community_data_file,
        checkpoint_file=community_articles_cp or COMMUNITY_ARTICLES_CHECKPOINT,
        limit=limit_community_articles,
        dry_run=dry_run,
    )
    community_2025 = filter_community_to_year(community_all, target_year)
    _write_year_output(community_output, "CIPD Community Blog", target_year, community_2025, timestamp)

    # 3) People Management pipeline
    people_links_file = CIPD_PEOPLE_LINKS_FILE
    people_data_file = CIPD_PEOPLE_DATA_FILE
    people_links_cp = None
    people_articles_cp = None
    if dry_run:
        os.makedirs(dry_run_dir, exist_ok=True)
        people_links_file = os.path.join(dry_run_dir, "scraped_links_people_management.json")
        people_data_file = os.path.join(dry_run_dir, "cipd_article_data_people_management.json")
        people_links_cp = os.path.join(dry_run_dir, "cp_cipd_people_links.json")
        people_articles_cp = os.path.join(dry_run_dir, "cp_cipd_people_articles.json")

    collect_people_links(
        max_pages=people_max_pages,
        output_file=people_links_file,
        checkpoint_file=people_links_cp or PEOPLE_LINKS_CHECKPOINT,
        dry_run=dry_run,
    )
    people_all = extract_people_articles(
        input_file=people_links_file,
        output_file=people_data_file,
        checkpoint_file=people_articles_cp or PEOPLE_ARTICLES_CHECKPOINT,
        limit=limit_people_articles,
        dry_run=dry_run,
    )
    people_2025 = filter_people_to_year(people_all, target_year)
    _write_year_output(people_output, "People Management", target_year, people_2025, timestamp)

    # 4) Podcast episode parity pipeline
    podcast_payload = {"total_series": 0, "total_episodes": 0, "series": []}
    podcast_year_payload = {
        "source": "CIPD Podcast Episodes",
        "year": target_year,
        "total_series": 0,
        "total_episodes": 0,
        "generated": timestamp,
        "series": [],
    }
    if include_podcast_stage:
        podcast_file = CIPD_PODCAST_EPISODES_FILE
        podcast_cp = None
        if dry_run:
            os.makedirs(dry_run_dir, exist_ok=True)
            podcast_file = os.path.join(dry_run_dir, "cipd_podcast_episode_data.json")
            podcast_cp = os.path.join(dry_run_dir, "cp_cipd_podcast_episodes.json")

        podcast_payload = scrape_podcast_episodes(
            output_file=podcast_file,
            checkpoint_file=podcast_cp or PODCAST_CHECKPOINT,
            max_pages=podcast_max_pages,
            dry_run=dry_run,
        )
        series_2025 = filter_podcast_to_year(podcast_payload.get("series", []), target_year)
        podcast_year_payload = {
            "source": "CIPD Podcast Episodes",
            "year": target_year,
            "total_series": len(series_2025),
            "total_episodes": sum(len(x.get("serie_podcast", [])) for x in series_2025),
            "generated": timestamp,
            "series": series_2025,
        }
        _save_json(podcast_output, podcast_year_payload)

    # 5) Coverage contract validation report
    coverage = build_coverage_report(
        target_year=target_year,
        knowledge_links_file=CIPD_ARTICLE_LINKS_FILE,
        knowledge_articles=knowledge_articles_all,
        community_articles=community_all,
        people_articles=people_all,
        podcast_payload=podcast_payload,
        include_podcast_stage=include_podcast_stage,
    )
    coverage["generated"] = timestamp
    coverage["outputs"] = {
        "knowledge": knowledge_output,
        "community": community_output,
        "people_management": people_output,
        "podcast_episodes": podcast_output if include_podcast_stage else "",
        "dry_run_artifacts": dry_run_dir if dry_run else "",
    }
    _save_json(coverage_output, coverage)

    summary = {
        "knowledge_output": knowledge_result.get("output_file", knowledge_output),
        "community_output": community_output,
        "people_output": people_output,
        "podcast_output": podcast_output if include_podcast_stage else "",
        "coverage_output": coverage_output,
        "coverage_summary": coverage.get("summary", {}),
    }
    return summary


def _build_cli():
    parser = argparse.ArgumentParser(description="Run unified CIPD 2025 pipeline with coverage validation.")
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument("--skip-podcast-episodes", action="store_true")
    parser.add_argument("--community-max-pages", type=int, default=None)
    parser.add_argument("--people-max-pages", type=int, default=None)
    parser.add_argument("--podcast-max-pages", type=int, default=50)
    parser.add_argument("--limit-community-articles", type=int, default=None)
    parser.add_argument("--limit-people-articles", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = _build_cli().parse_args()
    result = run_pipeline(
        target_year=args.target_year,
        include_podcast_stage=not args.skip_podcast_episodes,
        community_max_pages=args.community_max_pages,
        people_max_pages=args.people_max_pages,
        limit_community_articles=args.limit_community_articles,
        limit_people_articles=args.limit_people_articles,
        podcast_max_pages=args.podcast_max_pages,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
