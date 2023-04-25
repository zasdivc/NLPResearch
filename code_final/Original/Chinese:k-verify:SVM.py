#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 23:42:33 2023

@author: hangao
"""

# PCA
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import numpy as np
import re
import jieba
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.model_selection import cross_validate

# Read the CSV file and select the relevant columns
df = pd.read_csv('ChnSentiCorp_htl_all_translated.csv', usecols=['review', 'label'])
df['review'] = df['review'].astype(str)

# Define a function to keep only Chinese characters in a string
def keep_chinese(text):
    pattern = re.compile(r'[^\u4e00-\u9fff\s]')
    chinese_only = pattern.sub('', text)
    return chinese_only.strip()

# Tokenize the Chinese words and keep only the Chinese characters in each word
df['words-chinese'] = df['review'].apply(lambda x: [keep_chinese(word) for word in jieba.cut(x, cut_all=False) if keep_chinese(word) != ''])

# Train a Word2Vec model and calculate the mean of word embeddings for each tokenized sentence
word2vec_model = Word2Vec(sentences=df['words-chinese'], vector_size=100, window=5, min_count=1, workers=4)

def mean_word2vec(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Calculate the TF-IDF values for each tokenized sentence
vectorizer = TfidfVectorizer(use_idf=True)
X_tfidf = vectorizer.fit_transform(df['words-chinese'].apply(lambda x: ' '.join(x)))

# Combine the reshaped 'tf-idf' values and 'word2vec' values
X = np.hstack((X_tfidf.toarray(), np.vstack(df['words-chinese'].apply(lambda x: mean_word2vec(x, word2vec_model)))))

# Apply PCA to reduce the dimensionality of the features
pca = PCA(n_components=1000)
X_pca = pca.fit_transform(X)

# Print the percentage of variance explained by each principal component
print(pca.explained_variance_ratio_)

# Print the new shape of the feature matrix after applying PCA
print(X_pca.shape)


X = X_pca

ks = [5, 10]
results = []
for k in ks:
    svm_model = SVC(kernel='linear', random_state=42)
    scores = cross_validate(svm_model, X, df['label'].values, cv=k, scoring=('f1_weighted', 'recall_weighted', 'accuracy'))

    f1_scores = scores['test_f1_weighted']
    recall_scores = scores['test_recall_weighted']
    accuracy_scores = scores['test_accuracy']

    # Calculate and store the mean and standard deviation of F1 score, recall, and accuracy
    results.append((k, np.mean(f1_scores), np.std(f1_scores), np.mean(recall_scores), np.std(recall_scores), np.mean(accuracy_scores), np.std(accuracy_scores)))

# Print the results
print('K\tF1 Score (mean ± std)\tRecall (mean ± std)\tAccuracy (mean ± std)')
for result in results:
    print('{}\t{:.4f} ± {:.4f}\t\t{:.4f} ± {:.4f}\t\t{:.4f} ± {:.4f}'.format(result[0], result[1], result[2], result[3], result[4], result[5], result[6]))