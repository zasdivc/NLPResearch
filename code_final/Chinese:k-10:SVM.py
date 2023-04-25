#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 23:49:42 2023

@author: hangao
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, word2vec
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import numpy as np
import re
import jieba
from sklearn.decomposition import PCA

df_3500 = pd.read_csv('ChnSentiCorp_htl_first_5000_new.csv')
df_rest = pd.read_csv('ChnSentiCorp_htl_rest_new.csv')

df_rest = df_rest.dropna(subset=['words-chinese'])

sentences = df_3500['words-chinese']
stop_words = ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '上面', '下面', '前面', '后面', '中间', '不会', '没有', '可以', '因为', '所以', '但是', '然后', '现在', '一些', '这些', '那些', '这个', '那个', '他们', '她们', '我们', '你们', '就是', '可以', '就像', '一定', '一样', '这样', '那样', '不要', '不能', '需要', '一起', '一直', '一些', '一下', '吧', '着', '呢', '啊', '哦', '嗯', '唉', '嘛', '喔', '呀', '哈', '咳']
sentences = df_3500['words-chinese'].apply(lambda x: [word for word in x if word not in stop_words])

model = word2vec.Word2Vec(sentences=sentences, min_count=1, vector_size=10000, workers=4)

# Calculate the mean value of the non-zero elements in each row for TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_3500['words-chinese'])
tfidf_matrix_dense = tfidf_matrix.toarray()

# Replace zeros with NaN values to avoid division by zero
tfidf_matrix_dense[tfidf_matrix_dense == 0] = np.nan

# Calculate the mean value of the non-zero elements in each row for TF-IDF, replacing NaN values with the mean
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
tfidf_mean = imp_mean.fit_transform(tfidf_matrix_dense).sum(axis=1) / (tfidf_matrix_dense != 0).sum(axis=1)

# Calculate the mean value of the non-zero elements in each row for word embeddings
embedding_mean = []
for sentence in sentences:
    row_sum = np.zeros(10000)
    row_count = 0
    for word in sentence:
        if word in model.wv.key_to_index:
            row_sum += model.wv[word]
            row_count += 1
    if row_count > 0:
        row_mean = row_sum / row_count
    else:
        row_mean = np.zeros(10000)
    embedding_mean.append(row_mean)

# Add the new columns to the DataFrame
df_3500['tfidf_mean'] = tfidf_mean
df_3500['embedding_mean'] = embedding_mean

# Convert the columns to matrices
tfidf_matrix = np.vstack(df_3500['tfidf_mean'])

# ######### Feature Scaling #########
# scaler = MinMaxScaler()
# tfidf_matrix = scaler.fit_transform(tfidf_matrix)

embedding_matrix = np.vstack(df_3500['embedding_mean'])

# ######### Feature Scaling #########
# embedding_matrix = scaler.fit_transform(embedding_matrix)

X_5000 = np.concatenate((tfidf_matrix, embedding_matrix), axis=1)

# pca = PCA(n_components=1000)
# X_5000 = pca.fit_transform(X_5000)
# X_rest = pca.fit_transform(X_rest)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
f1_scores = []
recall_scores = []
accuracy_scores = []
precision_scores = []

svm_model = SVC(kernel='linear', random_state=42)

# maximum Entropy
# svm_model = LogisticRegression(max_iter=1000)

# svm_model = GradientBoostingClassifier()

for train_index, test_index in kf.split(X_5000):
    X_train, X_test = X_5000[train_index], X_5000[test_index]
    y_train, y_test = df_3500['label'].values[train_index], df_3500['label'].values[test_index]

    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
    accuracy_scores.append(accuracy_score(y_test, y_pred, normalize=True))
    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))

f1_mean = np.mean(f1_scores)
recall_mean = np.mean(recall_scores)
accuracy_mean = np.mean(accuracy_scores)
precision_mean = np.mean(precision_scores)

print(f"Metrics for SVM model with k-fold (k=10) on testing set (n=5000):\nF1 Score: {f1_mean:.4f}\nRecall: {recall_mean:.4f}\nAccuracy: {accuracy_mean:.4f}\nPrecision: {precision_mean:.4f}")


# Feature engineering for df_rest
sentences_rest = df_rest['words-chinese']
tfidf_matrix_rest = vectorizer.transform(sentences_rest)
tfidf_matrix_dense_rest = tfidf_matrix_rest.toarray()

# Replace zeros with NaN values to avoid division by zero
tfidf_matrix_dense_rest[tfidf_matrix_dense_rest == 0] = np.nan

# Calculate the mean value of the non-zero elements in each row for TF-IDF, replacing NaN values with the mean
tfidf_mean_rest = imp_mean.transform(tfidf_matrix_dense_rest).sum(axis=1) / (tfidf_matrix_dense_rest != 0).sum(axis=1)

embedding_mean_rest = []
for sentence in sentences_rest:
    row_sum = np.zeros(10000)
    row_count = 0
    for word in sentence:
        if word in model.wv.key_to_index:
            row_sum += model.wv[word]
            row_count += 1
    if row_count > 0:
        row_mean = row_sum / row_count
    else:
        row_mean = np.zeros(10000)
    embedding_mean_rest.append(row_mean)

# Add the new columns to the DataFrame
df_rest['tfidf_mean'] = tfidf_mean_rest
df_rest['embedding_mean'] = embedding_mean_rest

# Convert the columns to matrices
tfidf_matrix_rest = np.vstack(df_rest['tfidf_mean'])

# ######### Feature Scaling #########
# tfidf_matrix_rest = scaler.transform(tfidf_matrix_rest)

embedding_matrix_rest = np.vstack(df_rest['embedding_mean'])

# ######### Feature Scaling #########
# embedding_matrix_rest = scaler.transform(embedding_matrix_rest)

X_rest = np.concatenate((tfidf_matrix_rest, embedding_matrix_rest), axis=1)

y_pred_rest = svm_model.predict(X_rest)
f1_rest = f1_score(df_rest['label'].values, y_pred_rest, average='weighted')
recall_rest = recall_score(df_rest['label'].values, y_pred_rest, average='weighted')
accuracy_rest = accuracy_score(df_rest['label'].values, y_pred_rest, normalize=True)
precision_rest = precision_score(df_rest['label'].values, y_pred_rest, average='weighted')

print(f"\nMetrics for SVM model on remaining records (n={len(df_rest)}):\nF1 Score: {f1_rest:.4f}\nRecall: {recall_rest:.4f}\nAccuracy: {accuracy_rest:.4f}\nPrecision: {precision_rest:.4f}")