import pandas as pd
import re
import numpy as np
from rouge_score import rouge_scorer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
from pyvi import ViTokenizer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_excel_data(file_path):
    df = pd.read_excel(file_path)
    return df

def sentences_split(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def get_stopwords(file_stop_words):
    with open(file_stop_words, encoding='utf-8') as f:
        stop_words = f.read()
    stop_words = stop_words.split("\n")
    return stop_words

def process_text(text, stop_words):
    newString = text.lower()
    newString = re.sub(r'\(.*?\)', '', newString)
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\[.*?\]', '', newString)
    newString = re.sub(r'\{.*?\}', '', newString)
    newString = re.sub(r'"([^"]*)"', '', newString)
    newString = re.sub('"', '', newString)
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub(r'\.{2,}', '.', newString)    
    words = newString.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def sentencesVector(sentences, stop_words):
    w2v = KeyedVectors.load_word2vec_format("/home/phu/TextSummary/viet_vec/vi.vec")
    vocab = w2v.index_to_key 
    X = []
    for sentence in sentences:
        sentence_tokenized = ViTokenizer.tokenize(sentence)
        words = sentence_tokenized.split(" ")
        sentence_vec = np.zeros((100))
        for word in words:
            if word in vocab and word not in stop_words:
                sentence_vec += w2v[word]
        X.append(sentence_vec)
    return X

# def buildSummary(kmeans, X, sentences): 
#     n_clusters = 5
#     avg = [] 
#     for j in range(n_clusters):
#         idx = np.where(kmeans.labels_ == j)[0]
#         avg.append(np.mean(idx))
#     closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
#     ordering = sorted(range(n_clusters), key=lambda k: avg[k])
#     summary = ' '.join([sentences[closest[idx]] for idx in ordering])
#     return summary

def compute_similarity(original_text, summary_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original_text, summary_text])
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0][0]

def compute_rouge(summary, original_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original_summary, summary)
    return scores


def summarize_with_kmeans_tfidf(text, stop_words, n_clusters=3):

    text = process_text(text, stop_words)
    sentences = sentences_split(text)
    sentence_vectors = sentencesVector(sentences, stop_words)    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(sentence_vectors)
    avg = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_vectors)
    
    # Xếp thứ tự các cụm dựa trên độ trung bình của các chỉ số
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    
    # Tạo bản tóm tắt bằng cách kết hợp các câu đại diện
    summary = ' '.join([sentences[closest[idx]] for idx in ordering])
    
    return summary

def generate_summary(X, sentences, top_n, epochs, batch_size):
    X = np.array(X)  # Chuyển đổi danh sách thành mảng numpy
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape đầu vào thành (số lượng mẫu, độ dài câu, 1) để phù hợp với input của mô hình
    model = Sequential()
    model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))  # Input shape là (độ dài câu, 1)
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(X.shape[2]))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, X, epochs=epochs, batch_size=batch_size)  # Input và output đều là X
    sentence_representations = []
    for i in range(X.shape[0]):
        sentence_vec = np.reshape(X[i], (1, X.shape[1], X.shape[2]))  # Reshape để tương thích với input
        sentence_representation = model.predict(sentence_vec)[0]
        sentence_representations.append(sentence_representation)
    sentence_representations = np.array(sentence_representations)
    avg_representation = np.mean(sentence_representations, axis=0)  # Vector trung bình của tất cả các câu
    similarities = cosine_similarity([avg_representation], sentence_representations)[0]
    all_words = " ".join(sentences).split()    # Tính tần suất từ trong đoạn văn
    word_freq = Counter(all_words)
    adjusted_similarities = [] # Thêm tần suất từ vào độ tương đồng của các câu
    for i, sentence in enumerate(sentences):
        sentence_words = sentence.split()
        sentence_freq = sum(word_freq[word] for word in sentence_words)
        adjusted_similarities.append(similarities[i] + sentence_freq * 0.01)  # Cộng tần suất với hệ số
    top_sentence_indices = np.argsort(adjusted_similarities)[-top_n:][::-1]  # Chọn các câu có độ tương đồng cao nhất
    if 0 not in top_sentence_indices:
        top_sentence_indices = np.insert(top_sentence_indices, 0, 0)
    else:
        top_sentence_indices = np.concatenate(([0], top_sentence_indices[top_sentence_indices != 0]))
    selected_sentences = [sentences[idx] for idx in top_sentence_indices]
    summary = " ".join(selected_sentences)
    return summary

def summarization(contents, top_n_percentage):
    file_path = "/home/phu/TextSummary/filtered_stopwords.txt"
    stop_words = get_stopwords(file_path)
    text = process_text(contents,stop_words)
    
    sentences = sentences_split(text)
    print(sentences)
    X = sentencesVector(sentences, stop_words)
    
    top_n = int((len(sentences) * top_n_percentage) // 100)
    if top_n == 0:
        top_n = 1
    sum_lstm = generate_summary(X, sentences, top_n, epochs=30, batch_size=32)
    sentences_after_sum = sentences_split(sum_lstm)
    return sum_lstm

def generate_summary_h5(text):
    tokenizer = AutoTokenizer.from_pretrained("minhtoan/t5-small-vietnamese-news")  
    model = AutoModelForSeq2SeqLM.from_pretrained("minhtoan/t5-small-vietnamese-news")
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=600,  
        num_beams=10,     
        length_penalty=0.8,  
        temperature=0.9,  
        top_p=0.95,      
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary