# Vietnamese-Text-Summary-with-LSTM

Text summarization is the process of condensing a lengthy piece of text into a shorter version while retaining its main ideas and critical information. It is a key task in natural language processing (NLP) that helps users quickly grasp the essence of the content without reading the entire document.

![vietnamese text summary](image/Overview.png)

## Overview

This project focuses on developing automated Vietnamese text summarization methods using a combination of machine learning and deep learning techniques. The following key components are included:

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Evaluates the importance of words in a document relative to a collection of documents, helping to extract key terms for summarization.
- **Word2Vec**: A word embedding technique that represents words as continuous vectors, capturing their semantic meanings and relationships for improved contextual understanding.
- **KMeans Clustering**: Groups similar sentences, aiding in the extraction of representative content for summaries.
- **LSTM (Long Short-Term Memory)**: A deep learning model designed to process sequential data and generate coherent summaries.
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: A widely used metric for evaluating the quality of text summarization. It measures the overlap between the generated summary and the reference summary, providing scores such as ROUGE-1, ROUGE-2, and ROUGE-L.

## Dataset

The **Vietnamese Multiple Document Summarization (ViMs) Dataset** is used for this project. It consists of 300 clusters of news articles collected from the Vietnamese language version of Google News. The dataset was created with support from the Ho Chi Minh City Department of Science and Technology (Grant Number 15/2016/HĐ-SKHCN).

### Data Construction Process
- The dataset contains articles from five genres: world news, domestic news, business, entertainment, and sports.
- Each cluster contains 4 to 10 articles, with a total of 1,945 articles.
- Each article includes: title, content, source, publication date, author(s), tags, and headline summary.
- Two summaries were created for each cluster by two annotators using the MDSWriter system. The annotators are Vietnamese native speakers with some knowledge of natural language processing.

### Data Information
- **Original folder**: Contains 300 subdirectories (news clusters), with a total of 1,945 articles.
- **Summary folder**: Contains 300 subdirectories with 600 final summaries, two manual summaries per cluster.
- **S3_summary folder**: Contains 300 subdirectories with 600 "best sentence selection" summaries, marking important sentences with labels (1 for most important, 0 for others).

ViMs is useful for implementing and evaluating supervised machine learning models for Vietnamese abstractive multi-document summarization.

## Usage

This project includes a Flask web application that provides a user-friendly interface for Vietnamese text summarization. Follow these steps to run the application locally:

### Run the Flask Web App

1. Ensure you have Python 3.x installed.
2. Navigate to the project directory:
   ```bash
    cd text_apps

3. Navigate to the project directory:
   ```bash
    python apps.py

4. The Flask app will start running on http://127.0.0.1:5000/ by default.

![Python_Apps](images/SampleUI.png)

## References

1. **Building a Vector Space Model for Vietnamese**:
   - Source: [Viblo: Xây dựng mô hình không gian vector cho tiếng Việt](https://viblo.asia/p/xay-dung-mo-hinh-khong-gian-vector-cho-tieng-viet-GrLZDXr2Zk0)
   - This article provides an in-depth explanation of creating a vector space model for Vietnamese text, which is essential for text summarization and understanding word embeddings.

2. **ViMs Dataset**:
   - Source: [ViMs Dataset on Kaggle](https://www.kaggle.com/datasets/vtrnanh/sust-feature-data-new)
   - This dataset was used for training and evaluating the summarization model, containing a large collection of Vietnamese news articles organized into clusters with corresponding summaries.

3. **Word2Vec**:
   - Source: [Word2Vec Tutorial by Cô Ban](https://machinelearningcoban.com/tabml_book/ch_embedding/word2vec.html)
   - This tutorial provides a comprehensive introduction to the Word2Vec model, which is used for learning vector representations of words.

4. **LSTM (Long Short-Term Memory)**:
   - Source: [LSTM Explained on WebsiteHCM](https://websitehcm.com/long-short-term-memory-lstm-la-gi/)
   - This article offers an overview of Long Short-Term Memory (LSTM) networks and their application in deep learning for sequential data processing.

5. **LSTM (Video Tutorial)**:
   - Source: [LSTM Explanation Video on YouTube](https://www.youtube.com/watch?v=YcRPPy3EiJs&t=1953s&ab_channel=ProtonX)
   - This video provides a practical explanation of how LSTM networks work and their significance in handling time-series data and text summarization tasks.

6. **Word2Vec for Vietnamese**:
   - Source: [Word2Vec for Vietnamese on GitHub](https://github.com/sonvx/word2vecVN)
   - This GitHub repository contains a Word2Vec implementation tailored for the Vietnamese language, which was used to build word embeddings for Vietnamese text processing.

7. **Comprehensive Guide to Text Summarization**:
   - Source: [Analytics Vidhya Guide on Text Summarization](https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/)
   - This guide provides a thorough walkthrough of various techniques used in text summarization, including deep learning approaches such as LSTM and other methods to generate concise summaries.
