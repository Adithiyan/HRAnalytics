import os
import sys
import uuid

sys.path.insert(0, os.path.abspath("src"))

from cipd.community_extractor import extract_article as extract_community_article  # noqa: E402
from cipd.peoplemanagement_extractor import extract_article as extract_people_article  # noqa: E402
from cipd.podcast_episodes import scrape_podcast_episodes  # noqa: E402


def test_community_schema_dry_run():
    row = extract_community_article("https://example.com/community", dry_run=True)
    expected = {
        "url",
        "title",
        "author",
        "date",
        "year",
        "category",
        "tags",
        "description",
        "summary",
        "full_text",
        "links",
        "directory",
    }
    assert expected.issubset(set(row.keys()))


def test_peoplemanagement_schema_dry_run():
    row = extract_people_article("https://example.com/people", dry_run=True)
    expected = {
        "url",
        "title",
        "author",
        "date",
        "year",
        "category",
        "tags",
        "description",
        "summary",
        "full_text",
        "links",
        "directory",
    }
    assert expected.issubset(set(row.keys()))


def test_podcast_episode_schema_dry_run():
    case_dir = os.path.join(os.path.abspath("."), "tests_tmp", f"schema_{uuid.uuid4().hex}")
    os.makedirs(case_dir, exist_ok=True)
    out_file = os.path.join(case_dir, "podcast.json")
    cp_file = os.path.join(case_dir, "podcast_cp.json")
    payload = scrape_podcast_episodes(output_file=out_file, checkpoint_file=cp_file, dry_run=True)

    assert payload["series"]
    episode = payload["series"][0]["serie_podcast"][0]
    expected = {
        "podcast_title",
        "podcast_audio_link",
        "podcast_duration",
        "podcast_description",
        "podcast_transcript",
    }
    assert expected.issubset(set(episode.keys()))
