import re

import jieba
import nltk
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

stop_words = set(stopwords.words('english'))
stop_words.remove("very")
stop_words.remove("so")
stop_words.remove("above")
stop_words.remove("below")
stop_words.remove("up")
stop_words.remove("down")
stop_words.add("the")
stop_words_chinese = ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '上面', '下面', '前面', '后面', '中间', '不会', '没有', '可以', '因为', '所以', '但是', '然后', '现在', '一些', '这些', '那些', '这个', '那个', '他们', '她们', '我们', '你们', '就是', '可以', '就像', '一定', '一样', '这样', '那样', '不要', '不能', '需要', '一起', '一直', '一些', '一下', '吧', '着', '呢', '啊', '哦', '嗯', '唉', '嘛', '喔', '呀', '哈', '咳']


df = pd.read_csv('ChnSentiCorp_htl_all_translated.csv')
df['review'] = df['review'].apply(str)
df['translated_review'] = df['translated_review'].apply(str)

def keep_english(text):
    pattern = re.compile(r'[^a-zA-Z\s]')
    english_only = pattern.sub('', text)
    return english_only.strip().lower()

def keep_chinese(text):
    pattern = re.compile(r'[^\u4e00-\u9fff\s]')
    chinese_only = pattern.sub('', text)
    return chinese_only.strip()

def replace_punctuation_with_space(text):
    pattern = re.compile(r'[^\w\s]')
    return pattern.sub(' ', text)

# Replace punctuation with spaces
df['review'] = df['review'].apply(replace_punctuation_with_space)
df['translated_review'] = df['translated_review'].apply(replace_punctuation_with_space)

# Tokenize the Chinese words and keep only the Chinese characters in each word
df['words-chinese'] = df['review'].apply(lambda x: [keep_chinese(word) for word in jieba.cut(x, cut_all=False) if keep_chinese(word) != ''])

# Tokenize the English words and keep only the English characters in each word
df['words-english'] = df['translated_review'].apply(lambda x: [keep_english(word) for word in nltk.word_tokenize(x) if keep_english(word) != ''])

def remove_stopwords(text):
    return [word for word in text if word not in stop_words]

def remove_stopwords_chinese(text):
    return [word for word in text if word not in stop_words_chinese]

df['words-english'] = df['words-english'].apply(lambda x: remove_stopwords(x))
df['words-chinese'] = df['words-chinese'].apply(lambda x: remove_stopwords_chinese(x))

# Drop rows with no Chinese or English words
df = df[df['words-chinese'].apply(lambda x: len(x) > 0) & df['words-english'].apply(lambda x: len(x) > 0)]

wordCountEnglish = {}
for words in df['words-english']:
    for word in words:
        if word in wordCountEnglish:
            wordCountEnglish[word] += 1
        else:
            wordCountEnglish[word] = 1

wordCountChinese = {}
for words in df['words-chinese']:
    for word in words:
        if word in wordCountChinese:
            wordCountChinese[word] += 1
        else:
            wordCountChinese[word] = 1

ChineseCount = sorted(wordCountChinese.values(), reverse=True)
EnglishCount = sorted(wordCountEnglish.values(), reverse=True)

print("Chinese Count:", str(len(wordCountChinese)), str(ChineseCount))
print("English Count:", str(len(wordCountEnglish)), str(EnglishCount))

print("Top 10 English words:")
for i, (word, count) in enumerate(sorted(wordCountEnglish.items(), key=lambda x: x[1], reverse=True)[:10]):
    print(f"{i+1}. {word}: {count}")

print("\nTop 10 Chinese words:")
for i, (word, count) in enumerate(sorted(wordCountChinese.items(), key=lambda x: x[1], reverse=True)[:10]):
    print(f"{i+1}. {word}: {count}")
#
# plt.plot(list(wordCountEnglish.keys()), list(wordCountEnglish.values()))
# plt.xlabel('Word frequency')
# plt.ylabel('Number of words')
#
# plt.yscale('log')
# plt.title('English word frequency distribution')
# plt.show()

# df.to_csv('processed_reviews.csv', index=False)
#
# random_sample = df.sample(n=5000, random_state=42)
#
# random_sample.to_csv('ChnSentiCorp_htl_first_5000_new.csv', index=False)
#
# rest = df.drop(random_sample.index)
#
# rest.to_csv('ChnSentiCorp_htl_rest_new.csv', index=False)