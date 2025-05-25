#!/usr/bin/env python3

import json
import logging
import numpy as np
import pandas as pd
import spacy
from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import re
import argparse

# ------------------------------------------------------------------------------
# CONFIGURATION

DEFAULT_INPUT_FOLDER = r"test\newdata"

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
OUTPUT_SUFFIX = "_with_topics.json"

#INPUT_JSON_PATH = "test/data_src/cipd_article_links_people_management.json"
#OUTPUT_JSON_PATH = f"{INPUT_JSON_PATH}_with_topics.json"
NUM_TOPICS = 5
MAX_TOP_WORDS = 8
RANDOM_STATE = 7
LOG_LEVEL = logging.INFO

# ------------------------------------------------------------------------------
# LOGGING SETUP
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------
# NLP SETUP
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

CUSTOM_STOPWORDS = set([
    "hr", "hrm", "human", "resource", "resources", "workforce", "people",
    "personnel", "talent", "employees", "staff", "workers", "job", "jobs",
    "work", "organization", "organizations", "company", "employer", "employers",
    "employee", "employees", "workplace", "management", "administration",
    "profession", "im", "ive", "ill", "id", "youre", "youve", "youll", "youd",
    "hes", "shes", "its", "weve", "well", "wed", "theyre", "theyve", "theyll", "theyd",
    "thats", "whats", "whos", "wheres", "whens", "whys", "hows", "isnt", "arent",
    "wasnt", "werent", "havent", "hasnt", "hadnt", "wont", "wouldnt", "dont",
    "doesnt", "didnt", "cant", "couldnt", "shouldnt", "mustnt"
])

# ------------------------------------------------------------------------------
def load_json(json_path: str) -> tuple[int, pd.DataFrame]:
    logging.info("Loading JSON data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    total_articles = data.get("total_articles", 0)
    articles = data.get("articles", [])
    logging.info(f"Loaded {total_articles} articles.")
    return total_articles, pd.DataFrame(articles)

# ------------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    lines = text.strip().split('\n')
    social_tags = ("linkedin", "facebook", "youtube")
    while lines and any(tag in lines[-1].lower() for tag in social_tags):
        lines.pop()

    text = "\n".join(lines).lower()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and
              token.lemma_ not in CUSTOM_STOPWORDS and not token.is_stop]
    return " ".join(tokens)

# ------------------------------------------------------------------------------
def get_topic_labels(model, feature_names, n_top_words=10):
    topics = {}
    for idx, topic in enumerate(model.components_):
        top_features = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_features]
        topics[idx] = top_words
    return topics

# ------------------------------------------------------------------------------
def process_single_article(article_data: dict,
                          num_topics: int = NUM_TOPICS,
                          random_state: int = RANDOM_STATE) -> dict:
    combined_fields = []
    for field in ['title', 'category', 'tags', 'description', 'summary', 'full_text']:
        val = article_data.get(field)
        if isinstance(val, list):
            combined_fields.append(" ".join(val))
        elif isinstance(val, str):
            combined_fields.append(val)

    combined_text = ' '.join(combined_fields)
    cleaned_text = preprocess_text(combined_text)
    if not cleaned_text:
        return {
            "dominant_topic_label": None
        }

    #vectorizer = TfidfVectorizer()

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform([cleaned_text])


    lda = LatentDirichletAllocation(
    n_components=num_topics,
    random_state=random_state,
    max_iter=20,
    learning_method='online',
    learning_offset=10.0,
    doc_topic_prior=0.1,   # α: topic distribution per doc
    topic_word_prior=0.01  # β: word distribution per topic
    )

    lda.fit(X)

    topic_labels = get_topic_labels(lda, vectorizer.get_feature_names_out(), n_top_words=MAX_TOP_WORDS)
    doc_topic_distribution = lda.transform(X)[0]
    dominant_topic = int(np.argmax(doc_topic_distribution))

    return {
    #"dominant_topic_label": topic_labels[dominant_topic],
    "article_topics": list(topic_labels.values())
    }


# ------------------------------------------------------------------------------
def main():
    json_files = list(INPUT_FOLDER.glob("*.json"))
    #json_files = [INPUT_JSON_PATH]

    if not json_files:
        logging.warning("No JSON files found in the input folder.")
        return

    for json_file in json_files:
        logging.info(f"Processing file: {json_file}")
        try:
            total_articles, df = load_json(json_file)

            results = []
            for _, row in tqdm(df.iterrows(), total=len(df)):
                article_dict = row.to_dict()
                topic_data = process_single_article(article_dict)
                article_dict.update(topic_data)
                results.append(article_dict)

            # Clean NaNs
            for article in results:
                for key, value in article.items():
                    if isinstance(value, float) and np.isnan(value):
                        article[key] = None

            output_data = {
                "total_articles": len(results),
                "articles": results
            }

            output_file = json_file.with_name(json_file.stem + OUTPUT_SUFFIX)
            #output_file = str(json_file) + OUTPUT_SUFFIX
            with open(output_file, "w", encoding="utf-8") as out_f:
                raw_json = json.dumps(output_data, ensure_ascii=False, indent=2)

                def flatten_article_topics(match):
                    list_text = match.group(0)
                    flattened = re.sub(r'\s+', ' ', list_text).replace('\n', '').replace('  ', '')
                    return flattened

                processed_json = re.sub(
                    r'"article_topics":\s*\[\s*(?:\[[^\]]*\]\s*,?\s*)+\]',
                    flatten_article_topics,
                    raw_json
                )

                out_f.write(processed_json)

            logging.info(f"Saved output to: {output_file}")

        except Exception as e:
            logging.error(f"Failed to process {json_file}: {e}")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
