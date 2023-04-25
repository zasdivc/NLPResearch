from gensim.models import word2vec
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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
embedding_matrix = np.vstack(df_3500['embedding_mean'])
X = np.concatenate((tfidf_matrix, embedding_matrix), axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df_3500['label'], test_size=0.4, random_state=42)

# Train the SVM model
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the evaluation metrics
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print('test result:')

print('F1 score: {:.4f}'.format(f1))
print('Recall: {:.4f}'.format(recall))
print('Accuracy: {:.4f}'.format(accuracy))
print('precision: {:.4f}'.format(precision))


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
embedding_matrix_rest = np.vstack(df_rest['embedding_mean'])
X_rest = np.concatenate((tfidf_matrix_rest, embedding_matrix_rest), axis=1)

# Use the trained SVM model to predict on the new data
y_pred_rest = clf.predict(X_rest)



# Calculate the evaluation metrics on the new data
f1_rest = f1_score(df_rest['label'], y_pred_rest)
recall_rest = recall_score(df_rest['label'], y_pred_rest)
accuracy_rest = accuracy_score(df_rest['label'], y_pred_rest)
precision_rest = precision_score(df_rest['label'], y_pred_rest)

print('validate result:')
print('F1 score: {:.4f}'.format(f1_rest))
print('Recall: {:.4f}'.format(recall_rest))
print('Accuracy: {:.4f}'.format(accuracy_rest))
print('precision: {:.4f}'.format(precision_rest))

# df_3500.to_csv('ChnSentiCorp_htl_first_3500.csv', index=False)
# df_rest.to_csv('ChnSentiCorp_htl_rest.csv', index=False)
