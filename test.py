# PCA
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import numpy as np
import re
import jieba

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

# Split the data into training and testing sets and test different ratios
ratios = [0.7, 0.6]
results = []
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X, df['label'].values, test_size=(1-ratio), random_state=42)

    # Train the SVM classifier and predict on the testing set
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # Calculate and store the F1 score, recall, and accuracy
    results.append((ratio, f1_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), accuracy_score(y_test, y_pred)))

# Print the results
print('Train/Test Ratio\tF1 Score\tRecall\tAccuracy')
for result in results:
    print('{}\t{:.4f}\t\t{:.4f}\t{:.4f}'.format(result[0], result[1], result[2], result[3]))