# JSON to TXT Article Conversion Script
## THIS IS OUTDATED
## Overview
This script processes a JSON file containing articles and filters them based on keywords related to Human Resource Analytics. It then converts the selected articles into TXT files and stores metadata for tracking purposes.

## Features
- **Keyword-based filtering**: Searches for HR analytics-related terms in article metadata and content.
- **Hybrid Filtering Approach**:
  - **Exact keyword matching**
  - **Fuzzy keyword matching** (handles minor variations)
  - **NLP-based keyword extraction** (detects HR-related terms)
  - **TF-IDF & Cosine Similarity** (ensures semantic relevance)
- **Progress tracking**: Uses a progress bar (`tqdm`) for large datasets.
- **Metadata storage**: Saves details of selected articles in a JSON file for easy reference.

## Prerequisites
Before running the script, ensure you have the necessary Python libraries installed:

```bash
pip install tqdm rapidfuzz spacy scikit-learn nltk
python -m spacy download en_core_web_sm
```

## Usage
### 1. **Prepare the JSON Input File**
Ensure you have a JSON file containing articles. The expected format:
```json
{
    "articles": [
        {
            "title": "Introduction to HR Analytics",
            "category": "HR Technology",
            "tags": ["People Analytics", "HR Metrics"],
            "description": "Overview of HR Analytics in organizations.",
            "summary": "Using data to drive HR decisions.",
            "full_text": "Human Resource Analytics helps organizations..."
        }
    ]
}
```

### 2. **Run the Script**
Run the script in your terminal or command prompt:
```bash
python script.py
```

### 3. **Output Files**
- **Filtered Articles in TXT Format**: Saved in a dynamically named folder `filtered_articles_txt_<input_json>`
- **Metadata File (`00conversion_metadata.json`)**: Contains details about selected articles.

## How It Works
### **1. Keyword Matching**
The script searches for the presence of predefined HR-related keywords in different fields like title, category, tags, and content.

### **2. Fuzzy Matching**
Allows small variations of keywords (e.g., "HR Analytic" instead of "HR Analytics") using `rapidfuzz`.

### **3. NLP-Based Keyword Extraction**
Uses `spaCy` to extract **important nouns and named entities**, ensuring articles discussing HR analytics are included even if they use different terminology.

### **4. TF-IDF & Cosine Similarity**
Measures how semantically related an article is to HR Analytics terms using machine learning techniques.

## Configuration Options
- **`KEYWORDS`**: List of core HR Analytics terms.
- **`EXPANDED_KEYWORDS`**: Additional HR-related terms for a broader search.
- **`TF-IDF Threshold`**: Adjusts how closely an article must match HR-related terms.
- **`Fuzzy Matching Sensitivity`**: Can be increased/decreased to fine-tune relevance detection.

## Expected Console Output
```
🔍 Processing articles...
Processing Articles: 100%|███████████████████| 5000/5000 [00:30<00:00, 200/s]
✅ Conversion Complete!
📄 Total Articles Scanned: 5000
✅ Total Articles Selected: 250
📝 Metadata saved in: filtered_articles_txt_cipd_article_data_views_uk/00conversion_metadata.json
```

## Next Steps & Customization
- **Adjust Keywords & Thresholds**: Modify `KEYWORDS` and `EXPANDED_KEYWORDS` to fine-tune filtering.
- **Parallel Processing**: Can be added for handling very large datasets efficiently.
- **Integration with a Database**: Instead of saving TXT files, results can be stored in a structured database.

## Troubleshooting
1. **Issue: `OSError: [E050] Can't find model 'en_core_web_sm'`**
   - Solution: Run `python -m spacy download en_core_web_sm`

2. **Issue: Script runs but finds no matching articles**
   - Solution: Lower the `fuzzy_match` threshold and TF-IDF similarity threshold.


