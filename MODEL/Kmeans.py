import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class KMeansSummary:
    def __init__(self, processing_instance, n_clusters):
        self.processing = processing_instance
        self.n_clusters = n_clusters

    def summarize_with_kmeans(self, text, stop_words):  
        text = self.processing.process_text(text, stop_words)
        sentences = self.processing.sentences_split(text)
        
        sentence_vectors = self.processing.sentencesVector(sentences, stop_words)    
        kmeans = KMeans(n_clusters=self.n_clusters,random_state=42)
        kmeans.fit(sentence_vectors)
        avg = []
        for j in range(self.n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_vectors) # lấy ra khoảng cách gần nhất của vector so với trung tâm
        
        ordering = sorted(range(self.n_clusters), key=lambda k: avg[k])
        
        summary = ' '.join([sentences[closest[idx]] for idx in ordering])
        
        return summary
