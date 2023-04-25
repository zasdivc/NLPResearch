#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 23:29:24 2023

@author: hangao
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, accuracy_score
import numpy as np
import re
import jieba
from sklearn.decomposition import PCA

# Read the original CSV file
df = pd.read_csv('ChnSentiCorp_htl_all_translated.csv', usecols=['review', 'label'])

df_5000 = pd.read_csv('ChnSentiCorp_htl_first_5000.csv', usecols=['review', 'label'])


df_rest = pd.read_csv('ChnSentiCorp_htl_rest.csv', usecols=['review', 'label'])
df_rest = df_rest.dropna(subset=['review'])

# Define a function to keep only Chinese characters in a string
def keep_chinese(text):
    pattern = re.compile(r'[^\u4e00-\u9fff\s]')
    chinese_only = pattern.sub('', text)
    return chinese_only.strip()

# Tokenize the Chinese words and keep only the Chinese characters in each word
df_5000['words-chinese'] = df_5000['review'].apply(lambda x: [keep_chinese(word) for word in jieba.cut(x, cut_all=False) if keep_chinese(word) != ''])
df_rest['words-chinese'] = df_rest['review'].apply(lambda x: [keep_chinese(word) for word in jieba.cut(x, cut_all=False) if keep_chinese(word) != ''])

df_rest = df_rest.dropna(subset=['review'])

# Train a Word2Vec model and calculate the mean of word embeddings for each tokenized sentence
word2vec_model = Word2Vec(sentences=df_5000['words-chinese'], vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.build_vocab(df_rest['words-chinese'], update=True)
word2vec_model.train(df_rest['words-chinese'], total_examples=word2vec_model.corpus_count, epochs=word2vec_model.epochs)

# Calculate the TF-IDF values for each tokenized sentence
vectorizer = TfidfVectorizer(use_idf=True)
X_tfidf_5000 = vectorizer.fit_transform(df_5000['words-chinese'].apply(lambda x: ' '.join(x)))
X_tfidf_rest = vectorizer.transform(df_rest['words-chinese'].apply(lambda x: ' '.join(x)))

def mean_word2vec(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

X_5000 = np.hstack((X_tfidf_5000.toarray(), np.vstack(df_5000['words-chinese'].apply(lambda x: mean_word2vec(x, word2vec_model)))))
X_rest = np.hstack((X_tfidf_rest.toarray(), np.vstack(df_rest['words-chinese'].apply(lambda x: mean_word2vec(x, word2vec_model)))))

pca = PCA(n_components=1000)
X_5000 = pca.fit_transform(X_5000)
X_rest = pca.fit_transform(X_rest)



X_train, X_test, y_train, y_test = train_test_split(X_5000, df_5000['label'].values, test_size=(1 - 0.7), random_state=42)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)


f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
print(f"Metrics for SVM model on testing set (n=1500):\nF1 Score: {f1:.4f}\nRecall: {recall:.4f}\nAccuracy: {accuracy:.4f}")

y_pred_rest = svm_model.predict(X_rest)
f1_rest = f1_score(df_rest['label'].values, y_pred_rest, average='weighted')
recall_rest = recall_score(df_rest['label'].values, y_pred_rest, average='weighted')
accuracy_rest = accuracy_score(df_rest['label'].values, y_pred_rest)
print(f"\nMetrics for SVM model on remaining records (n={len(df_rest)}):\nF1 Score: {f1_rest:.4f}\nRecall: {recall_rest:.4f}\nAccuracy: {accuracy_rest:.4f}")
