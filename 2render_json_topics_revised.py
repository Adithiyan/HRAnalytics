#!/usr/bin/env python3

import json
import os
import re
import logging
from pathlib import Path
from datetime import datetime
import argparse
# --------------------------------------------------------------------

# Argument parsing

DEFAULT_INPUT_FOLDER = r"newdata"

parser = argparse.ArgumentParser(description="Analyze all JSON article files in a folder.")
parser.add_argument(
    "input_folder",
    nargs="?",
    default=DEFAULT_INPUT_FOLDER,
    type=str,
    help="Path to the folder containing JSON files. Defaults to a preset folder if not provided."
)
args = parser.parse_args()

INPUT_FOLDER = Path(args.input_folder)

ORIGINAL_INPUT_FILENAME = "SHRM_FinalOutput_News_with_topics"
#INPUT_JSON_PATH = "test/newdata/SHRM_FinalOutput_News_with_topics.json"
OUTPUT_JSON_PATH = f"test/output/filtered_articles_{ORIGINAL_INPUT_FILENAME}.json"
OUTPUT_TXT_DIR = Path("output")
METADATA_FILE = os.path.join(OUTPUT_TXT_DIR, "00conversion_metadata.json")
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------------------------
# KEYWORDS FOR FILTERING
CONCEPT_KEYWORDS = [
    "analytics", "analytics", "data", "metrics", "ai", "intelligence", "insights",
    "model", "modeling", "models",
    "algorithm", "algorithmic", "algorithms",
    "evidencebased", "predictive"
]


def keyword_match(topic_word_lists, concept_keywords=None) -> bool:
    target_keywords = {"analytic", "analytics"}
    if not topic_word_lists:
        return False
    if all(isinstance(w, str) for w in topic_word_lists):
        all_words = {w.lower() for w in topic_word_lists}
    elif all(isinstance(w, list) for w in topic_word_lists):
        all_words = {w.lower() for sub in topic_word_lists for w in sub}
    else:
        return False
    return any(kw in all_words for kw in target_keywords)



def sanitize_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "_", title)

def extract_date(date_string):
    formats = [
        "%d %b, %Y",    # 22 May, 2025
        "%B %d, %Y",    # February 1, 2005
        "%b %d, %Y",    # May 22, 2025
        "%d %B %Y",     # 21 February 2017
        "%Y-%m-%d",     # 2025-05-22
        "%Y/%m/%d",     # 2025/05/22
        "%d/%m/%Y",     # 22/05/2025
        "%m/%d/%Y",     # 05/22/2025
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    logging.warning(f"⚠️ Unrecognized date format: {date_string}")
    return "unknown-date"


def format_filename(title, date):
    date_part = extract_date(date)
    title_part = sanitize_filename(title).replace(" ", "-")[:80]
    return f"{date_part}-{title_part}.txt"

def remove_trailing_social_lines(text):
    if not isinstance(text, str):
        return ""
    lines = text.strip().split('\n')
    pattern = re.compile(r'(?i)(linkedin|facebook|youtube)')
    while lines and pattern.search(lines[-1]):
        lines.pop()
    return "\n".join(lines)



# --------------------------------------------------------------------
def extract_text_fields(article: dict) -> str:
    parts = []
    if article.get("title"):
        parts.append(f"TITLE: {article['title']}")
    if article.get("summary"):
        parts.append(f"\nSUMMARY:\n{article['summary']}")
    if article.get("full_text"):
        clean_text = remove_trailing_social_lines(article['full_text'])
        parts.append(f"\nFULL TEXT:\n{clean_text}")
    return "\n\n".join(parts)

def save_txt_and_metadata(articles, output_dir, total_scanned, metadata_file):
    os.makedirs(output_dir, exist_ok=True)
    metadata_list = []

    for article in articles:
        title = article.get("title", "untitled")
        date = article.get("date", "unknown-date")
        filename = format_filename(title, date)
        filepath = os.path.join(output_dir, filename)

        metadata = {
            "Title": title,
            "Date": date,
            "URL": article.get("url", "N/A"),
            "Category": article.get("category", "N/A"),
            "Author": article.get("author", "N/A"),
            "Tags": article.get("tags", []),
            "DominantTopicLabel": article.get("dominant_topic_label"),
            "ArticleTopics": article.get("article_topics"),
            "File": filename,
            "Description": article.get("description", "N/A"),
            "Directory": article.get("directory", "N/A"),
            "FilterPassed": True,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(metadata, indent=4))
            f.write("\n\n")
            f.write(article.get("full_text", "No content available."))

        metadata_list.append(metadata)

    # Save metadata summary JSON
    with open(metadata_file, "w", encoding="utf-8") as meta_file:
        json.dump({
            "total_articles_scanned": total_scanned,
            "total_articles_selected": len(articles),
            "selected_articles": metadata_list
        }, meta_file, indent=4)

# --------------------------------------------------------------------
def main():
    input_files = list(INPUT_FOLDER.glob("*_with_topics.json"))
    #input_files = [Path(INPUT_JSON_PATH)]
    if not input_files:
        logging.warning("No *_with_topics.json files found in input folder.")
        return

    for json_file in input_files:
        logging.info(f"Processing: {json_file}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            articles_all = data.get("articles", [])
            selected = []

            for article in articles_all:
                keywords_raw = article.get("article_topics")
                if not keywords_raw:
                    continue
                if isinstance(keywords_raw, str):
                    try:
                        keywords = json.loads(keywords_raw)
                    except json.JSONDecodeError:
                        continue
                else:
                    keywords = keywords_raw

                if keyword_match(keywords):
                    selected.append(article)

            logging.info(f"✅ {len(selected)} matched out of {len(articles_all)}")

            
            base_name = json_file.stem.replace("_with_topics", "")
            output_json_path = OUTPUT_TXT_DIR / f"filtered_articles_{base_name}.json"
            output_txt_dir = OUTPUT_TXT_DIR / f"relevant_articles_txt_{base_name}"

            metadata_file = output_txt_dir / "00conversion_metadata.json"

            # Then pass `metadata_file` into save_txt_and_metadata:
            save_txt_and_metadata(selected, output_txt_dir, len(articles_all), metadata_file)


            with open(output_json_path, "w", encoding="utf-8") as f_out:
                json.dump({"filtered_articles": selected}, f_out, ensure_ascii=False, indent=2)

            save_txt_and_metadata(selected, output_txt_dir, len(articles_all), metadata_file)

            logging.info(f"📝 Saved metadata to {metadata_file}")
            logging.info(f"📁 Saved filtered JSON to {output_json_path}")
            logging.info(f"📄 Exported text files to: {output_txt_dir}")

        except Exception as e:
            logging.error(f"Failed to process {json_file}: {e}")


# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
