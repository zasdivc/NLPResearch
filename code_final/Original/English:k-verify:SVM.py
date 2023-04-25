#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:00:39 2023

@author: hangao
"""

# English k-fold ratios
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the data
df = pd.read_csv('ChnSentiCorp_htl_all_translated.csv')
df['review'] = df['review'].apply(str)
df['translated_review'] = df['translated_review'].apply(str)

# Define a function to keep only English characters in a string
def keep_english(text):
    pattern = re.compile(r'[^a-zA-Z\s]')
    english_only = pattern.sub('', text)
    return english_only.strip()

# Tokenize the English words and keep only the English characters in each word
df['words-english'] = df['translated_review'].apply(lambda x: [keep_english(word) for word in nltk.word_tokenize(x) if keep_english(word) != ''])

# Calculate the mean of TF-IDF values for each tokenized sentence
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['words-english'].apply(lambda x: ' '.join(x)))
df['tf-idf'] = np.mean(X_tfidf.toarray(), axis=1)

# Train a Word2Vec model and calculate the mean of word embeddings for each tokenized sentence
word2vec_model = Word2Vec(sentences=df['words-english'], vector_size=100, window=5, min_count=1, workers=4)

def mean_word2vec(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

df['word2vec'] = df['words-english'].apply(lambda x: mean_word2vec(x, word2vec_model))

# Combine the reshaped 'tf-idf' values and 'word2vec' values
X = np.hstack((df['tf-idf'].values.reshape(-1, 1), np.vstack(df['word2vec'].values)))
y = df['label'].values

print(X.shape)

# Apply PCA to reduce the dimensionality of the features
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

# Print the percentage of variance explained by each principal component
print(pca.explained_variance_ratio_)

# Print the new shape of the feature matrix after applying PCA
print(X_pca.shape)

X = X_pca

# Define the k-fold split ratios
kfolds = [5, 10]

# Initialize lists to store performance metrics for each ratio
f1_scores = []
recall_scores = []
accuracy_scores = []
precision_scores = []
results = []

# Iterate over the kfolds
for kfold in kfolds:
    # Split the data into training and testing sets using the current kfold ratio
    kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_train, y_train)
    
        # Predict on the testing set
        y_pred = svm_model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        
        f1_scores.append(f1)
        recall_scores.append(recall)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        results.append((kfold, f1, recall, accuracy, precision))

print(f'K-Fold\t\tF1 Score\tRecall\t\tAccuracy\tPrecision')
for i in range(len(kfolds)):
    print(f'{kfolds[i]:d}\t\t{f1_scores[i]:.4f}\t\t{recall_scores[i]:.4f}\t\t{accuracy_scores[i]:.4f}\t\t{precision_scores[i]:.4f}')
            