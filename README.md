# HR Research Data Pipeline - CIPD & SHRM (2025)

This repository contains the current `src/` scraping pipelines plus legacy scripts used earlier for CIPD and SHRM collection.

## Current Status

- CIPD parity work is implemented in `src/` (knowledge + community + people management + podcast episode parity + coverage contract report).
- SHRM parity check is complete: the `src/shrm/` pipeline matches the legacy SHRM page surfaces and selectors.

## Legacy-to-src Parity

### CIPD

Legacy sources reviewed:
- `legacy/scripts/cipd.py`
- `cipd/cipd1_mainlinks.py`
- `cipd/cipd2_articlelinks.py`
- `cipd/cipd3_articles.py`
- `cipd/cipd4_articlelinksplpmgt.py`
- `cipd/cipd4_articlesplpmgt.py`
- `cipd/cipd5_articlelinksblog.py`
- `cipd/cipd5_articlesblog*.py`

`src/` coverage now includes:
- Knowledge hub scraping (all legacy sections):
  - `factsheets`, `guides`, `reports`, `tools`, `podcasts`, `webinars`,
    `case-studies`, `evidence-reviews`, `bitesize-research`, `employment-law`
- Community blog link + article scraping
- People Management link + article scraping
- Podcast episode-level parity output (`podcast_audio_link`, `podcast_duration`, `podcast_transcript`, etc.)
- Coverage validator report with statuses:
  - `covered`, `covered_zero_2025`, `missing`

### SHRM

Legacy sources reviewed:
- `legacy/scripts/shrm1.py`
- `legacy/scripts/shrm2.py`
- `legacy/scripts/shrm2optimized.py`

Parity result:
- `src/shrm/link_collector.py` matches legacy SHRM link collection surface:
  - Start page: `https://www.shrm.org/topics-tools`
  - Pagination: `atomic-pager` next button in shadow DOM
  - Link extraction: `atomic-result` shadow DOM anchors
- `src/shrm/article_extractor.py` matches legacy SHRM article extraction surface:
  - Same page-level extraction model from collected SHRM article URLs
  - Same core selectors (`h1.content__title`, `span.content__date`, `span.content__author`, breadcrumb, tags, `div.cmp-text`)
- `src/phase3_shrm_links.py` and `src/phase4_shrm_articles.py` add checkpoint/resume and year filtering on top of equivalent scraping behavior.

## Main Pipelines

### Unified CIPD (recommended)

```bash
python src/run_cipd_2025.py
```

Options:

```bash
python src/run_cipd_2025.py --skip-podcast-episodes
python src/run_cipd_2025.py --target-year 2025
python src/run_cipd_2025.py --dry-run --community-max-pages 2 --people-max-pages 2 --podcast-max-pages 1
```

Outputs (in `output/`):
- `cipd_2025_<timestamp>.json`
- `cipd_community_2025_<timestamp>.json`
- `cipd_people_management_2025_<timestamp>.json`
- `cipd_podcast_episodes_2025_<timestamp>.json` (unless skipped)
- `cipd_coverage_<timestamp>.json`

### SHRM phased run

```bash
python src/phase3_shrm_links.py
python src/phase4_shrm_articles.py
```

Phase 3 debug examples:

```bash
# short smoke run with fresh checkpoint + visible browser
python src/phase3_shrm_links.py --max-pages 20 --clear-checkpoint

# headless run with tighter transition timeout
python src/phase3_shrm_links.py --max-pages 50 --headless --transition-timeout 8
```

If pagination gets stuck, Phase 3 now writes:
- `output/shrm_debug_stuck_page_<n>.html`
- `output/shrm_debug_stuck_page_<n>.png`

Final SHRM 2025 output:
- `output/shrm_2025_<timestamp>.json`
- `output/shrm_coverage_<timestamp>.json`

### Combined CIPD+SHRM run

```bash
python src/run_2025.py
```

## Checkpoint Files

- `src/checkpoints/cipd_links.json`
- `src/checkpoints/cipd_articles.json`
- `src/checkpoints/cipd_community_links.json`
- `src/checkpoints/cipd_community_articles.json`
- `src/checkpoints/cipd_people_links.json`
- `src/checkpoints/cipd_people_articles.json`
- `src/checkpoints/cipd_podcast_episodes.json`
- `src/checkpoints/shrm_links.txt`
- `src/checkpoints/shrm_articles.json`

## Tests

Parity and coverage tests:

```bash
python -m pytest -q -p no:cacheprovider tests/test_cipd_parity_contract.py tests/test_cipd_coverage_validator.py tests/test_cipd_stage_smoke.py tests/test_cipd_output_schema.py tests/test_shrm_parity_contract.py tests/test_shrm_coverage_validator.py
```

## Requirements

- Python 3.10+
- Playwright + Chromium
- Dependencies from your environment (`requests`, `beautifulsoup4`, `playwright`, etc.)
- Credentials configured in `src/config.py` and `misc/.env` (for SHRM)
