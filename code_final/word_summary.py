import pandas as pd

df = pd.read_csv('processed_reviews.csv')

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