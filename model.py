import json
import os
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import bigrams, trigrams
from utils import setup_logger
from pandarallel import pandarallel


logger = setup_logger("keyword_extractor")

class KeyWordExtractor:
    def __init__(self, keywords_per_tag: int = 10):
        self.keywords_per_tag = keywords_per_tag
    
    def train(self, df_train: pd.DataFrame, save_model_path: str):
        """Extract keywords for each tag and save it"""
        logger.info("Starting the training.")

        if os.path.isfile(save_model_path):
            logger.info("Keywords for each tag are already extracted, skipping training")
            return

        tag_to_keywords = defaultdict(list)
        # Since we can have several tags per text
        df_train['tags'] = df_train['tags'].str.split(',')
        df_train = df_train.explode('tags')
        
        tag_to_text = df_train.groupby('tags')['text'].agg(list).to_dict()
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        for tag, texts in tag_to_text.items():
            tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            indexed_scores = sorted(zip(tfidf_scores, feature_names), reverse=True)
            tag_to_keywords[tag] = [word for _, word in indexed_scores[:self.keywords_per_tag]]
        
        # Save the most common tag
        tag_to_keywords["MOST_COMMON_TAG"] = df_train["tags"].value_counts().keys()[0]

        with open(save_model_path, "w") as f:
            json.dump(tag_to_keywords, f)

    def evaluate(self, df_test: pd.DataFrame, model_path: str, overlap_threshold: float):
        """
        Evaluate calculated tag keywords on the test data
        
        We will use the following algorithm for detecting tags.
        1. If there are one or more tags higher than threshold, we take those as predictions
        2. If there are no tags higher than threshold, we will take one tag with highest overlap
        3. If there is no overlap with any of the tags we return the majority tag in the dataset.
        
        """
        logger.info("Extracting tags from test set")
        with open(model_path, "r") as f:
            tag_to_keywords = json.load(f)
        
        most_common_tag = tag_to_keywords["MOST_COMMON_TAG"]
        del tag_to_keywords["MOST_COMMON_TAG"]

        def extract_grams(text: str):
            """Convert text into a set of words, bigrams and trigrams"""
            tokens = text.split()
            words = set(tokens)
            bigram_set = set(' '.join(bigram) for bigram in bigrams(tokens))
            trigram_set = set(' '.join(trigram) for trigram in trigrams(tokens))
            keyword_set = words.union(bigram_set).union(trigram_set)
            return keyword_set

        def filter_labels(text_tokens: set):
            """Detect the tags related to the text based on overlap"""

            detected_tags = []

            top_overlap, top_tag = 0, most_common_tag
            for tag in tag_to_keywords:
                keywords = set(tag_to_keywords[tag])
                overlap = len(text_tokens & keywords) / len(keywords)
                if overlap > overlap_threshold:
                    detected_tags.append(tag)
                elif overlap > top_overlap:
                    top_overlap = overlap
                    top_tag = tag
            
            if not detected_tags:
                return [top_tag]
            else:
                return detected_tags
        
        true = df_test["tags"].apply(lambda x: x.split(",")).tolist()
        preds = df_test["text"].parallel_apply(lambda x: filter_labels(extract_grams(x))).tolist()

        return true, preds
        

        

    