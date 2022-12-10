# Program to filter out first 10k reviews
# Filter out first 2000 reviews for each star
import json
import random

def add_to_files(file_path, array):
    with open(file_path, "w") as fd:
        json.dump(array, fd, indent=2)
    fd.close()

def filter_stars():
    total_reviews = 0

    one_star = []
    two_star = []
    three_star = []
    four_star = []
    five_star = []

    one_star_ids = []
    two_star_ids = []
    three_star_ids = []
    four_star_ids = []
    five_star_ids = []

    restaurant_fd =  open("./json_data/restaurant_ids.json", "r")
    restaurant_ids = json.dumps(json.load(restaurant_fd))
    restaurant_fd.close()

    reviews_fd = open("./yelp_academic_dataset_review.json", "r")

    for line in reviews_fd:
        review = json.loads(line)
        total_reviews += 1

        # Append to array by star
        # Get 2000 reviews for each category to get a total of 10k reviews
        if review['business_id'] in restaurant_ids:
            if review['stars'] == 1.0 and len(one_star_ids) < 2000:
                one_star.append(review)
                one_star_ids.append(review["review_id"])
            if review['stars'] == 2.0 and len(two_star_ids) < 2000:
                two_star.append(review)
                two_star_ids.append(review["review_id"])
            if review['stars'] == 3.0 and len(three_star_ids) < 2000:
                three_star.append(review)
                three_star_ids.append(review["review_id"])
            if review['stars'] == 4.0 and len(four_star_ids) < 2000:
                four_star.append(review)
                four_star_ids.append(review["review_id"])
            if review['stars'] == 5.0 and len(five_star_ids) < 2000:
                five_star.append(review)
                five_star_ids.append(review["review_id"])
            # print("one_star", len(one_star_ids), "two_star", len(two_star_ids), "three_star", len(three_star_ids), "four_star", len(four_star_ids), "five_star", len(five_star_ids))
            if len(one_star_ids) == 2000 and len(two_star_ids) == 2000 and len(three_star_ids) == 2000 and len(four_star_ids) == 2000 and len(five_star_ids) == 2000:
                break
    reviews_fd.close()

    # Shuffle Reviews before splitting to testing and training sets
    random.shuffle(one_star)
    random.shuffle(two_star)
    random.shuffle(three_star)
    random.shuffle(four_star)
    random.shuffle(five_star)

    # 90% for Training 
    add_to_files("./json_data/first_10k/train/first_10k_1_star.json", one_star[200:])
    add_to_files("./json_data/first_10k/train/first_10k_2_star.json", two_star[200:])
    add_to_files("./json_data/first_10k/train/first_10k_3_star.json", three_star[200:])
    add_to_files("./json_data/first_10k/train/first_10k_4_star.json", four_star[200:])
    add_to_files("./json_data/first_10k/train/first_10k_5_star.json", five_star[200:])

    add_to_files("./json_data/first_10k/train/first_10k_review_ids_1_star_test.json", one_star_ids[200:])
    add_to_files("./json_data/first_10k/train/first_10k_review_ids_2_star_test.json", two_star_ids[200:])
    add_to_files("./json_data/first_10k/train/first_10k_review_ids_3_star_test.json", three_star_ids[200:])
    add_to_files("./json_data/first_10k/train/first_10k_review_ids_4_star_test.json", four_star_ids[200:])
    add_to_files("./json_data/first_10k/train/first_10k_review_ids_5_star_test.json", five_star_ids[200:])

    # 10% for Testing
    add_to_files("./json_data/first_10k/test/first_10k_1_star_test.json", one_star[:200])
    add_to_files("./json_data/first_10k/test/first_10k_2_star_test.json", two_star[:200])
    add_to_files("./json_data/first_10k/test/first_10k_3_star_test.json", three_star[:200])
    add_to_files("./json_data/first_10k/test/first_10k_4_star_test.json", four_star[:200])
    add_to_files("./json_data/first_10k/test/first_10k_5_star_test.json", five_star[:200])

    add_to_files("./json_data/first_10k/test/first_10k_review_ids_1_star.json", one_star_ids[:200])
    add_to_files("./json_data/first_10k/test/first_10k_review_ids_2_star.json", two_star_ids[:200])
    add_to_files("./json_data/first_10k/test/first_10k_review_ids_3_star.json", three_star_ids[:200])
    add_to_files("./json_data/first_10k/test/first_10k_review_ids_4_star.json", four_star_ids[:200])
    add_to_files("./json_data/first_10k/test/first_10k_review_ids_5_star.json", five_star_ids[:200])


    print(total_reviews)
    print(len(one_star_ids), len(two_star_ids), len(three_star_ids), len(four_star_ids), len(five_star_ids))

filter_stars()