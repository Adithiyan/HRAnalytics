import json
import os
import re
import subprocess
from tqdm import tqdm  # Progress Bar
from rapidfuzz import fuzz
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Ensure necessary NLTK data is downloaded
nltk.download("stopwords")

# Load NLP model for keyword extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Configuration
KEYWORDS = ["Human Resource Analytics", "HR Analytics", "People Analytics", "Workforce Analytics"]

EXPANDED_KEYWORDS = KEYWORDS + [
    "HR Data", "Employee Analytics", "Workforce Insights", "Talent Analytics",
    "HR Metrics", "People Insights", "Workforce Planning", "Organizational Analytics"
]  

INPUT_JSON = "cipd_article_data_views_uk.json"
OUTPUT_DIR = f"filtered_articles_txt_{INPUT_JSON}"
METADATA_FILE = f"{OUTPUT_DIR}\\00conversion_metadata.json"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize TF-IDF vectorizer with n-grams
vectorizer = TfidfVectorizer(
    stop_words="english", 
    ngram_range=(1, 3)  # Includes unigrams, bigrams, and trigrams
)

# Initialize stemming and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def sanitize_filename(title):
    """Sanitize the title to create a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', "_", title)

def preprocess_text(text):
    """Lowercases, removes stopwords, and applies stemming."""
    words = text.lower().split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def fuzzy_match(text, keywords, threshold=70):
    """Fuzzy matching with a lower threshold for inclusiveness."""
    text = text.lower()
    for keyword in keywords:
        if fuzz.partial_ratio(keyword.lower(), text) > threshold:
            return True
    return False

def extract_keywords(text):
    """Extract nouns and named entities from the text using NLP."""
    doc = nlp(text.lower())
    return {token.text for token in doc if token.pos_ in {"NOUN", "PROPN"}}  # Extract only nouns & proper nouns

def tfidf_similarity(text):
    """Check semantic similarity using TF-IDF with n-grams."""
    processed_text = preprocess_text(text)
    documents = [processed_text] + EXPANDED_KEYWORDS  # First document is the article, others are keywords
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])  # Compare article with keywords
    return any(score > 0.3 for score in similarity_scores[0])  # Lowered threshold for inclusivity

def contains_keyword_advanced(article):
    """Hybrid filtering approach combining multiple methods."""
    fields_to_check = [
        article.get("title", ""),
        article.get("category", ""),
        " ".join(article.get("tags", [])),
        article.get("description", ""),
        article.get("summary", ""),
        article.get("full_text", "")[:500]  # Take first 500 characters for keyword check
    ]
    
    text_to_search = " ".join(fields_to_check).lower()

    # 1. Basic String Matching (with expanded keywords)
    if any(keyword.lower() in text_to_search for keyword in EXPANDED_KEYWORDS):
        return True

    # 2. Fuzzy Matching
    if fuzzy_match(text_to_search, EXPANDED_KEYWORDS):
        return True

    # 3. NLP Keyword Extraction
    extracted_keywords = extract_keywords(text_to_search)
    if any(keyword.lower() in extracted_keywords for keyword in EXPANDED_KEYWORDS):
        return True

    # 4. TF-IDF Similarity Check
    if tfidf_similarity(text_to_search):
        return True

    return False

def convert_json_to_txt(json_file):
    """Convert JSON articles to TXT files using advanced filtering and log metadata."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    articles = data.get("articles", [])
    selected_articles = []
    
    print("\n🔍 Processing articles...\n")
    
    # Using tqdm for progress tracking
    for article in tqdm(articles, desc="Processing Articles", unit="article"):
        if contains_keyword_advanced(article):
            title = article.get("title", "untitled")
            sanitized_title = sanitize_filename(title) + ".txt"
            file_path = os.path.join(OUTPUT_DIR, sanitized_title)
            
            metadata = {
                "Title": article.get("title", "N/A"),
                "URL": article.get("url", "N/A"),
                "Category": article.get("category", "N/A"),
                "Date": article.get("date", "N/A"),
                "Author": article.get("author", "N/A"),
                "Tags": article.get("tags", []),
                "File": sanitized_title
            }
            
            full_text = article.get("full_text", "No content available.")
            
            # Save article as TXT
            with open(file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(json.dumps(metadata, indent=4) + "\n\n")
                txt_file.write(full_text)

            selected_articles.append(metadata)

    # Save metadata JSON file
    metadata_summary = {
        "total_articles_scanned": len(articles),
        "total_articles_selected": len(selected_articles),
        "selected_articles": selected_articles
    }
    
    with open(METADATA_FILE, "w", encoding="utf-8") as meta_file:
        json.dump(metadata_summary, meta_file, indent=4)

    # Display results
    print(f"\n✅ Conversion Complete!")
    print(f"📄 Total Articles Scanned: {len(articles)}")
    print(f"✅ Total Articles Selected: {len(selected_articles)}")
    print(f"📝 Metadata saved in: {METADATA_FILE}")

# Run the script
convert_json_to_txt(INPUT_JSON)
