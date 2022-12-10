# Program to find sentence and word count
import nltk
import json
import re

word_count = 0
sent_count = 0
file_names = ["./json_data/first_10k/train/first_10k_1_star.json",
"./json_data/first_10k/train/first_10k_2_star.json",
"./json_data/first_10k/train/first_10k_3_star.json",
"./json_data/first_10k/train/first_10k_4_star.json",
"./json_data/first_10k/train/first_10k_5_star.json",
"./json_data/first_10k/test/first_10k_1_star_test.json",
"./json_data/first_10k/test/first_10k_2_star_test.json",
"./json_data/first_10k/test/first_10k_3_star_test.json",
"./json_data/first_10k/test/first_10k_4_star_test.json",
"./json_data/first_10k/test/first_10k_5_star_test.json",
]


for file in file_names:
    with open(file, 'r', encoding='utf-8') as restaurant_fd:
        data = json.loads(restaurant_fd.read())
        for review in data:
            review["text"] = re.sub("\n\n", " ", review["text"])
            review["text"] = re.sub("\n", " ", review["text"])
            sent = review["text"].split()
            for word in sent:
                word_count += 1
            sent_count += 1
    restaurant_fd.close()

print(word_count)
print(sent_count)