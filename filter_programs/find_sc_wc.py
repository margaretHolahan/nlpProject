# Program to find all sentence and word count in our corpus
import nltk
import json

sent_count = 0
word_count = 0

fd = open("json_data/first_10k/all_reviews.json", "r", encoding='utf-8')
data = json.loads(fd.read())
for review in data:
    all_sents = nltk.sent_tokenize(review["text"])
    for sent in all_sents:
        sent_count += 1
        for word in sent:
            word_count +=1

print("Sentence Count", sent_count)
print("Word Count", word_count)

# Sentence Count 78588
# Word Count 5568695