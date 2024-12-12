import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from pyvi import ViTokenizer
import nltk

class LSTMModel:
    def __init__(self, processing_instance, top_n_percentage=30, epochs=30, batch_size=32):
        self.processing = processing_instance
        self.top_n_percentage = top_n_percentage
        self.epochs = epochs
        self.batch_size = batch_size

    def generate_summary(self, X, sentences, top_n):
        X = np.array(X)  # Chuyển đổi danh sách thành mảng numpy
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape đầu vào thành (số lượng mẫu, độ dài câu, 1) để phù hợp với input của mô hình
        model = Sequential() 
        model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))  # Input shape là (độ dài câu, feature)
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(X.shape[2]))
        model.compile(loss='mse', optimizer='adam')
        model.fit(X, X, epochs=self.epochs, batch_size=self.batch_size)  # Input và output đều là X
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
        adjusted_similarities = []  # Thêm tần suất từ vào độ tương đồng của các câu
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

    def summarization(self, contents):
        file_path = "/home/phu/TextSummary/filtered_stopwords.txt"
        stop_words = self.processing.get_stopwords(file_path)
        text = self.processing.process_text(contents,stop_words)
        
        sentences = self.processing.sentences_split(text)
        X = self.processing.sentencesVector(sentences,stop_words)
        
        top_n = int((len(sentences) * self.top_n_percentage) // 100)
        if top_n == 0:
            top_n = 1
        sum_lstm = self.generate_summary(X, sentences, top_n)
        sentences_after_sum = self.processing.sentences_split(sum_lstm)
        return sum_lstm
