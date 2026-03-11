"""
phase3_shrm_links.py — SHRM article link collection with checkpoint/resume.

Navigates SHRM's Topics & Tools hub (newest articles first), collecting
article URLs from Coveo atomic-result shadow DOM components.

Checkpoint: src/checkpoints/shrm_links.txt
  - One URL per line; appended after every page.
  - On restart, already-collected URLs are loaded and pagination resumes
    from the correct page number.

Output: output/shrm_links_2025run_<timestamp>.txt

Run:
    cd E:\\Adi\\Work\\RA\\Yao\\main
    python src/phase3_shrm_links.py
"""

import os
import sys
import time
import random
import argparse
from datetime import datetime
from playwright.sync_api import sync_playwright

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from config import OUTPUT_DIR, SHRM_TOPICS_URL
from shrm.link_collector import handle_cookies

CHECKPOINT_FILE = os.path.join(SRC_DIR, "checkpoints", "shrm_links.txt")
MAX_PAGES       = 400   # ~4,000 most-recent articles; covers all of 2025
RETRIES         = 3
TRACE_SAMPLES   = 3
MAX_CONSECUTIVE_ZERO_NEW = 15
STUCK_TRANSITION_LIMIT   = 3

os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_checkpoint():
    """Return (seen_links set, all_links list) from checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"[Checkpoint] Loaded {len(lines)} links from previous run.")
        return set(lines), lines
    return set(), []


def append_checkpoint(new_links):
    """Append new links to the checkpoint file."""
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        for link in new_links:
            f.write(link + "\n")


def extract_visible_and_new_links(page, seen_links, all_links):
    """
    Return (visible_links, new_on_page) where:
      - visible_links: all hrefs visible on the current page
      - new_on_page: hrefs not previously seen (also appended to all_links)
    """
    results = page.query_selector_all('atomic-result')
    visible_links = []
    new_on_page = []

    for result in results:
        try:
            shadow = result.evaluate_handle('el => el.shadowRoot')
            link = shadow.eval_on_selector('a', 'el => el.href')
            if link and link.strip():
                link = link.strip()
                visible_links.append(link)
                if link not in seen_links:
                    seen_links.add(link)
                    all_links.append(link)
                    new_on_page.append(link)
        except Exception as e:
            print(f"  Link error: {e}")

    return visible_links, new_on_page


def get_pager_state(page):
    """
    Return (pager_exists, next_enabled, current_page_label).
    """
    pager = page.query_selector('atomic-pager')
    if not pager:
        return False, False, ""

    shadow = pager.evaluate_handle('el => el.shadowRoot')
    next_enabled = bool(
        shadow.eval_on_selector('button[aria-label=\"Next\"]', 'el => !!el && !el.disabled')
    )

    current_page_label = ""
    try:
        current_page_label = shadow.eval_on_selector(
            'button[aria-current=\"true\"]',
            'el => el ? el.textContent.trim() : \"\"'
        ) or ""
    except Exception:
        current_page_label = ""

    return True, next_enabled, current_page_label


def wait_for_transition(page, before_fingerprint, before_label, timeout_seconds=10):
    """
    Wait until either result fingerprint or pager label changes.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        visible_links, _ = extract_visible_and_new_links(page, set(), [])
        after_fingerprint = "|".join(visible_links[:10])
        _, _, after_label = get_pager_state(page)

        if after_fingerprint != before_fingerprint or after_label != before_label:
            return True, after_fingerprint, after_label

        time.sleep(0.5)

    return False, before_fingerprint, before_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect SHRM article links with checkpoint/resume and pagination tracing."
    )
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument("--retries", type=int, default=RETRIES)
    parser.add_argument("--trace-samples", type=int, default=TRACE_SAMPLES)
    parser.add_argument("--transition-timeout", type=float, default=10.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--clear-checkpoint", action="store_true")
    return parser.parse_args()


def _clip(text, limit=180):
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "..."


def main():
    args = parse_args()

    if args.clear_checkpoint and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"[Checkpoint] Cleared existing file: {CHECKPOINT_FILE}")

    print("=" * 60)
    print("  Phase 3 — SHRM Link Collection")
    print(f"  Max pages: {args.max_pages}")
    print(f"  Retries/page: {args.retries}")
    print(f"  Headless: {args.headless}")
    print(f"  Transition timeout: {args.transition_timeout}s")
    print(f"  Checkpoint: {CHECKPOINT_FILE}")
    print("=" * 60)

    seen_links, all_links = load_checkpoint()
    start_page = 1  # Always start from page 1 — deduplication handles overlaps

    print(f"Starting collection (deduplication active). "
          f"Already have {len(all_links)} unique links.")
    if all_links:
        print("Note: non-empty checkpoint means many pages may show 0 new links on reruns.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page    = browser.new_page()

        print(f"Navigating to {SHRM_TOPICS_URL}")
        page.goto(SHRM_TOPICS_URL)
        handle_cookies(page)

        consecutive_zero_new = 0
        stuck_transitions = 0

        for current_page in range(start_page, args.max_pages + 1):
            print(f"\n--- Page {current_page}/{args.max_pages} "
                  f"(total links so far: {len(all_links)}) ---")
            success = False

            for attempt in range(args.retries):
                try:
                    page.wait_for_selector('atomic-result', timeout=20000)
                    visible_links, new_on_page = extract_visible_and_new_links(page, seen_links, all_links)
                    page_fingerprint = "|".join(visible_links[:10])
                    pager_exists, next_enabled, page_label = get_pager_state(page)
                    print(
                        f"  visible_results={len(visible_links)} "
                        f"new_links={len(new_on_page)} "
                        f"pager_exists={pager_exists} next_enabled={next_enabled} "
                        f"pager_page='{page_label}'"
                    )
                    if visible_links:
                        print(f"  sample_links={visible_links[:args.trace_samples]}")
                        print(f"  page_fingerprint={_clip(page_fingerprint)}")
                    print(f"  current_url={page.url}")

                    # Save new links to checkpoint immediately
                    if new_on_page:
                        append_checkpoint(new_on_page)
                        consecutive_zero_new = 0
                    else:
                        consecutive_zero_new += 1
                        print(f"  consecutive_zero_new_pages={consecutive_zero_new}")
                        if consecutive_zero_new >= MAX_CONSECUTIVE_ZERO_NEW:
                            print(
                                f"  Hit {MAX_CONSECUTIVE_ZERO_NEW} consecutive pages with 0 new links. "
                                "Stopping early to avoid looping on stale results."
                            )
                            success = True
                            break

                    # Advance to next page
                    if not pager_exists:
                        print("  Pager not found.")
                        success = True
                        break

                    if not next_enabled:
                        print("  No more pages (next disabled).")
                        success = True
                        break

                    # Click next and verify actual page transition.
                    pager = page.query_selector('atomic-pager')
                    shadow = pager.evaluate_handle('el => el.shadowRoot')
                    print(f"  click_next from_pager_page='{page_label}'")
                    shadow.eval_on_selector('button[aria-label="Next"]', 'el => el.click()')

                    changed, _, new_label = wait_for_transition(
                        page,
                        before_fingerprint=page_fingerprint,
                        before_label=page_label,
                        timeout_seconds=args.transition_timeout,
                    )

                    if changed:
                        stuck_transitions = 0
                        print(
                            f"  transition_ok=True from_pager_page='{page_label}' "
                            f"to_pager_page='{new_label}'"
                        )
                        time.sleep(random.uniform(0.8, 1.6))
                    else:
                        stuck_transitions += 1
                        print(
                            "  transition_ok=False (results/page marker unchanged after click) "
                            f"stuck_transitions={stuck_transitions}"
                        )
                        debug_html = os.path.join(OUTPUT_DIR, f"shrm_debug_stuck_page_{current_page}.html")
                        with open(debug_html, "w", encoding="utf-8") as f:
                            f.write(page.content())
                        print(f"  wrote_debug_html={debug_html}")
                        debug_png = os.path.join(OUTPUT_DIR, f"shrm_debug_stuck_page_{current_page}.png")
                        try:
                            page.screenshot(path=debug_png, full_page=True)
                            print(f"  wrote_debug_screenshot={debug_png}")
                        except Exception as screenshot_error:
                            print(f"  screenshot_error={screenshot_error}")
                        if stuck_transitions >= STUCK_TRANSITION_LIMIT:
                            print("  Pagination appears stuck repeatedly. Stopping early.")
                            success = True
                            break
                        # retry this page loop
                        continue

                    success = True
                    break

                except Exception as e:
                    print(f"  Attempt {attempt+1} failed: {e}")
                    time.sleep(2)

            if not success:
                print(f"  Page {current_page} failed after {args.retries} retries. Stopping.")
                break

        browser.close()

    # Write final output file
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"shrm_links_2025run_{timestamp}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_links))

    print(f"\n{'='*60}")
    print(f"Phase 3 complete.")
    print(f"  Total unique links : {len(all_links)}")
    print(f"  Output             : {output_file}")
    print(f"  Checkpoint         : {CHECKPOINT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
