import json
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath("src"))

from cipd.community_extractor import extract_all as extract_community_articles  # noqa: E402
from cipd.community_links import collect_links as collect_community_links  # noqa: E402
from cipd.peoplemanagement_extractor import extract_all as extract_people_articles  # noqa: E402
from cipd.peoplemanagement_links import collect_links as collect_people_links  # noqa: E402
from cipd.podcast_episodes import scrape_podcast_episodes  # noqa: E402


def _case_dir(name):
    path = os.path.join(os.path.abspath("."), "tests_tmp", f"{name}_{uuid.uuid4().hex}")
    os.makedirs(path, exist_ok=True)
    return path


def test_community_links_resume_dry_run():
    case_dir = _case_dir("community_links")
    out_file = os.path.join(case_dir, "community_links.json")
    cp_file = os.path.join(case_dir, "community_links_cp.json")

    links_first = collect_community_links(
        max_pages=3,
        output_file=out_file,
        checkpoint_file=cp_file,
        dry_run=True,
    )
    links_second = collect_community_links(
        max_pages=3,
        output_file=out_file,
        checkpoint_file=cp_file,
        dry_run=True,
    )
    assert len(links_first) == 3
    assert len(links_second) == 3
    assert os.path.exists(cp_file)


def test_people_links_resume_dry_run():
    case_dir = _case_dir("people_links")
    out_file = os.path.join(case_dir, "people_links.json")
    cp_file = os.path.join(case_dir, "people_links_cp.json")

    links_first = collect_people_links(
        max_pages=3,
        output_file=out_file,
        checkpoint_file=cp_file,
        dry_run=True,
    )
    links_second = collect_people_links(
        max_pages=3,
        output_file=out_file,
        checkpoint_file=cp_file,
        dry_run=True,
    )
    assert len(links_first) == 3
    assert len(links_second) == 3
    assert os.path.exists(cp_file)


def test_extractors_resume_dry_run():
    case_dir = _case_dir("extractors")
    links_file = os.path.join(case_dir, "links.json")
    links_payload = {"total_articles": 2, "articles": ["https://example.com/a", "https://example.com/b"]}
    with open(links_file, "w", encoding="utf-8") as f:
        json.dump(links_payload, f)

    # Community extractor
    c_out = os.path.join(case_dir, "community_articles.json")
    c_cp = os.path.join(case_dir, "community_articles_cp.json")
    first = extract_community_articles(
        input_file=links_file,
        output_file=c_out,
        checkpoint_file=c_cp,
        dry_run=True,
    )
    second = extract_community_articles(
        input_file=links_file,
        output_file=c_out,
        checkpoint_file=c_cp,
        dry_run=True,
    )
    assert len(first) == 2
    assert len(second) == 2

    # People extractor
    p_out = os.path.join(case_dir, "people_articles.json")
    p_cp = os.path.join(case_dir, "people_articles_cp.json")
    first_p = extract_people_articles(
        input_file=links_file,
        output_file=p_out,
        checkpoint_file=p_cp,
        dry_run=True,
    )
    second_p = extract_people_articles(
        input_file=links_file,
        output_file=p_out,
        checkpoint_file=p_cp,
        dry_run=True,
    )
    assert len(first_p) == 2
    assert len(second_p) == 2


def test_podcast_stage_resume_dry_run():
    case_dir = _case_dir("podcast")
    out_file = os.path.join(case_dir, "podcast.json")
    cp_file = os.path.join(case_dir, "podcast_cp.json")

    first = scrape_podcast_episodes(
        output_file=out_file,
        checkpoint_file=cp_file,
        dry_run=True,
    )
    second = scrape_podcast_episodes(
        output_file=out_file,
        checkpoint_file=cp_file,
        dry_run=True,
    )

    assert first["total_series"] == 1
    assert first["total_episodes"] == 1
    assert second["total_series"] == 1
    assert second["total_episodes"] == 1
