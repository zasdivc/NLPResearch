#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:09:29 2023

@author: hangao
"""

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

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the data
df = pd.read_csv('ChnSentiCorp_htl_first_5000.csv')
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

# Define the number of folds
n_splits = 10

# Split the data into training and testing sets using k-fold cross-validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize arrays to store evaluation metrics for each fold
f1_scores = np.zeros(n_splits)
recall_scores = np.zeros(n_splits)
accuracy_scores = np.zeros(n_splits)
precision_scores = np.zeros(n_splits)

svm_model = SVC(kernel='linear', random_state=42)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i+1}")

    # Split the data into training and testing sets for the current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the SVM classifier
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    # Predict on the testing set
    y_pred_test = svm_model.predict(X_test)

    # Calculate evaluation metrics for the testing set
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted')

    # Store the evaluation metrics for the current fold
    f1_scores[i] = f1_test
    recall_scores[i] = recall_test
    accuracy_scores[i] = accuracy_test
    precision_scores[i] = precision_test

    # Print the evaluation metrics for the current fold
    print(f'Testing set performance:\nF1 score: {f1_test:.4f}\nRecall: {recall_test:.4f}\nAccuracy: {accuracy_test:.4f}\nPrecision: {precision_test:.4f}\n')

# Print the mean and standard deviation of the evaluation metrics across all folds
print(f'Average testing set performance across {n_splits} folds:\nF1 score: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}\nRecall: {np.mean(recall_scores):.4f} +/- {np.std(recall_scores):.4f}\nAccuracy: {np.mean(accuracy_scores):.4f} +/- {np.std(accuracy_scores):.4f}\nPrecision: {np.mean(precision_scores):.4f} +/- {np.std(precision_scores):.4f}\n')


df_rest = pd.read_csv('ChnSentiCorp_htl_rest.csv')
df_rest['translated_review'] = df_rest['translated_review'].apply(str)

df_rest['words-english'] = df_rest['translated_review'].apply(lambda x: nltk.word_tokenize(x))

df_rest['words-english'] = df_rest['words-english'].apply(lambda x: [word for word in x if not keep_english(word)])

X_tfidf_rest = vectorizer.transform(df_rest['words-english'].apply(lambda x: ' '.join(x)))
df_rest['tf-idf'] = np.mean(X_tfidf_rest.toarray(), axis=1)

df_rest['word2vec'] = df_rest['words-english'].apply(lambda x: mean_word2vec(x, word2vec_model))

X_rest = np.hstack((df_rest['tf-idf'].values.reshape(-1, 1), np.vstack(df_rest['word2vec'].values)))
y_rest = df_rest['label'].values

X_pca_rest = pca.transform(X_rest)
X_rest = X_pca_rest

y_pred_rest = svm_model.predict(X_rest)

f1_rest = f1_score(y_rest, y_pred_rest, average='weighted')
recall_rest = recall_score(y_rest, y_pred_rest, average='weighted')
accuracy_rest = accuracy_score(y_rest, y_pred_rest)
precision_rest = precision_score(y_rest, y_pred_rest, average='weighted')

print(f'Rest of the dataset performance:\nF1 score: {f1_rest:.4f}\nRecall: {recall_rest:.4f}\nAccuracy: {accuracy_rest:.4f}\nPrecision: {precision_rest:.4f}')