import pandas as pd
import re
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
from pyvi import ViTokenizer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.feature_extraction.text import TfidfVectorizer

class Processing:
    def __init__(self, stop_words_file, w2v_model_path):
        self.stop_words = self.get_stopwords(stop_words_file)
        self.w2v = KeyedVectors.load_word2vec_format(w2v_model_path)
        self.vocab = self.w2v.index_to_key

    def load_excel_data(self, file_path):
        df = pd.read_excel(file_path)
        return df

    def sentences_split(self, text):
        sentences = nltk.sent_tokenize(text)
        return sentences

    def get_stopwords(self, file_stop_words):
        with open(file_stop_words, encoding='utf-8') as f:
            stop_words = f.read()
        stop_words = stop_words.split("\n")
        return stop_words

    def process_text(self,text, stop_words):
        newString = text.lower()
        newString = re.sub(r'\(.*?\)', '', newString)
        newString = BeautifulSoup(newString, "lxml").text
        newString = re.sub(r'\[.*?\]', '', newString)
        newString = re.sub(r'\{.*?\}', '', newString)
        newString = re.sub(r'"([^"]*)"', '', newString)
        newString = re.sub('"', '', newString)
        newString = re.sub(r'\.[a-zA-Z]{2,6}\b', '', newString)
        # newString = re.sub(r'.*?http[^\s]*', '', newString)
        # newString = re.sub(r'\([^)]*\)', '', newString)
        newString = re.sub(r'\.{2,}', '.', newString)    
        words = newString.split()
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)

    def sentencesVector(self, sentences,stop_words):
        X = []
        for sentence in sentences:
            sentence_tokenized = ViTokenizer.tokenize(sentence)
            words = sentence_tokenized.split(" ")
            sentence_vec = np.zeros((100))
            for word in words:
                if word in self.vocab and word not in stop_words:
                    sentence_vec += self.w2v[word]
            X.append(sentence_vec)
        return X

    def compute_similarity(self, original_text, summary_text):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([original_text, summary_text])
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity_matrix[0][0]

    def compute_rouge(self, summary, original_summary):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(original_summary, summary)
        return scores
