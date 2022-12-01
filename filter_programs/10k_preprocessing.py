# Program to collect various information about all files
import json
import re

def process_files(json_files, file_name):
    for i in range(len(json_files)):
        with open(json_files[i], 'r', encoding='utf-8') as star_fd:
            print(json_files[i])
            data = json.loads(star_fd.read())
            reviews = []
            for review in data:
                review["text"] = review["text"].lower()
                review["text"] = re.sub("\n", " ", review["text"])
                review["text"] = re.sub("(['.,\";?!--()*/:~)])", r' \1 ', review["text"])
                reviews.append(review)
            with open(file_name[i], 'w', encoding='utf-8') as new_fd:
                json.dump(reviews, new_fd, indent=2)
            new_fd.close()
        star_fd.close()

def main():
    json_files = ["./json_data/first_10k/train/first_10k_1_star.json",
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

    file_name = ["./json_data/tokenize/first_10k/train/first_10k_1_star.json",
    "./json_data/tokenize/first_10k/train/first_10k_2_star.json",
    "./json_data/tokenize/first_10k/train/first_10k_3_star.json",
    "./json_data/tokenize/first_10k/train/first_10k_4_star.json",
    "./json_data/tokenize/first_10k/train/first_10k_5_star.json",
    "./json_data/tokenize/first_10k/test/first_10k_1_star_test.json",
    "./json_data/tokenize/first_10k/test/first_10k_2_star_test.json",
    "./json_data/tokenize/first_10k/test/first_10k_3_star_test.json",
    "./json_data/tokenize/first_10k/test/first_10k_4_star_test.json",
    "./json_data/tokenize/first_10k/test/first_10k_5_star_test.json",
    ]
    process_files(json_files, file_name)

main()