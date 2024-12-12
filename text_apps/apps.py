import importlib
import sys
import os
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
from rouge_score import rouge_scorer

# Add the path to your project to sys.path
sys.path.append('/home/phu/TextSummary')

# Dynamically import the required modules
KMeansSummary = importlib.import_module("MODEL.Kmeans").KMeansSummary
LSTMModel = importlib.import_module("MODEL.LSTM").LSTMModel
Processing = importlib.import_module("MODEL.processing").Processing

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentence_count = word_count = summary_count = words_sum = summary_count_kmeans = words_sum_kmeans = origin_summary_sent_count = words_origin_count = 0
    summary = summary_kmeans = ''
    original_summary = ''
    similarity_score = rouge1_score_lstm = rouge2_score_lstm = rouge_score_lstm = 0.0
    rouge1_score_kmeans = rouge2_score_kmeans = rouge_score_kmeans = 0.0
    default_text = "Đây là đoạn văn mặc định để hiển thị khi không có dữ liệu nhập."
    input_text = default_text
    img_base64 = ''  # Đảm bảo img_base64 luôn được khởi tạo

    if request.method == 'POST':
        input_text = request.form.get('input_text', default_text)
        top_n = int(request.form.get('top_n', 40))  # Mặc định là 40 nếu không có giá trị
        top_n_kmeans = 3
        # Khởi tạo instance của các class
        processing_instance = Processing("/home/phu/TextSummary/filtered_stopwords.txt","/home/phu/TextSummary/viet_vec/vi.vec")
        kmeans_summary = KMeansSummary(processing_instance, n_clusters=top_n_kmeans)
        lstm_model = LSTMModel(processing_instance, top_n_percentage=top_n, epochs=30, batch_size=32)

        # Tóm tắt bằng mô hình LSTM
        summary = lstm_model.summarization(input_text)

        # Tóm tắt bằng KMeans-TFIDF
        stop_words = []  # Hoặc sử dụng danh sách stop words thích hợp
        summary_kmeans = kmeans_summary.summarize_with_kmeans(input_text, stop_words)

        # Đếm số câu và từ trong các bản tóm tắt
        sentence_count = len(input_text.split('.'))
        word_count = len(input_text.split())
        summary_count = len(summary.split('.'))
        words_sum = len(summary.split())
        summary_count_kmeans = len(summary_kmeans.split('.'))
        words_sum_kmeans = len(summary_kmeans.split())

        # So sánh điểm ROUGE giữa các phương pháp
        file_path = '/home/phu/TextSummary/NLP/dataset/dataset.xlsx'
        df = processing_instance.load_excel_data(file_path)
        best_match = None
        highest_similarity = 0.0
        similarity_threshold = 0.4

        for index, row in df.iterrows():
            origin_text = row['Text']
            similarity = processing_instance.compute_similarity(input_text, origin_text)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = row

        if best_match is not None and highest_similarity >= similarity_threshold:
            original_summary = best_match['Summary']
            similarity_score = processing_instance.compute_similarity(summary, original_summary)
            origin_summary_sent_count = len(original_summary.split('.'))
            words_origin_count = len(original_summary.split())
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

            # Điểm ROUGE cho LSTM
            rouge_scores_lstm = scorer.score(original_summary, summary)
            rouge_score_lstm = rouge_scores_lstm['rougeL'].fmeasure
            rouge1_score_lstm = rouge_scores_lstm['rouge1'].fmeasure
            rouge2_score_lstm = rouge_scores_lstm['rouge2'].fmeasure

            # Điểm ROUGE cho KMeans
            rouge_scores_kmeans = scorer.score(original_summary, summary_kmeans)
            rouge_score_kmeans = rouge_scores_kmeans['rougeL'].fmeasure
            rouge1_score_kmeans = rouge_scores_kmeans['rouge1'].fmeasure
            rouge2_score_kmeans = rouge_scores_kmeans['rouge2'].fmeasure

        else:
            original_summary = "Không có bản tóm tắt gốc."

        # Vẽ biểu đồ so sánh điểm ROUGE
        labels = ['LSTM', 'KMeans']
        rouge1_scores = [rouge1_score_lstm, rouge1_score_kmeans]
        rouge2_scores = [rouge2_score_lstm, rouge2_score_kmeans]
        rouge_scores = [rouge_score_lstm, rouge_score_kmeans]

        fig, ax = plt.subplots(figsize=(6, 4))
        width = 0.2
        x = range(len(labels))

        ax.bar([p - width for p in x], rouge1_scores, width=width, label='ROUGE-1')
        ax.bar(x, rouge2_scores, width=width, label='ROUGE-2')
        ax.bar([p + width for p in x], rouge_scores, width=width, label='ROUGE-L')

        ax.set_ylabel('ROUGE Score')
        ax.set_title('So sánh điểm ROUGE giữa LSTM và KMeans')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        img = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()

    return render_template('index.html',
                           input_text=input_text,
                           sentence_count=sentence_count,
                           word_count=word_count,
                           summary_count=summary_count,
                           words_sum=words_sum,
                           summary=summary,
                           summary_kmeans=summary_kmeans,
                           summary_count_kmeans=summary_count_kmeans,
                           words_sum_kmeans=words_sum_kmeans,
                           original_summary=original_summary,
                           similarity_score=similarity_score,
                           rouge_score_lstm=rouge_score_lstm,
                           rouge1_score_lstm=rouge1_score_lstm,
                           rouge2_score_lstm=rouge2_score_lstm,
                           rouge_score_kmeans=rouge_score_kmeans,
                           rouge1_score_kmeans=rouge1_score_kmeans,
                           rouge2_score_kmeans=rouge2_score_kmeans,
                           origin_summary_sent_count=origin_summary_sent_count,
                           words_origin_count=words_origin_count,
                           img_base64=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
