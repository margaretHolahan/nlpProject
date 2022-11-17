import json

total_reviews = 0
our_reviews = []
review_ids = []

restaurant_fd =  open("json_data/restaurant_ids.json", "r")
restaurant_ids = json.dumps(json.load(restaurant_fd))
restaurant_fd.close()

users_fd =  open("json_data/top3_user_ids.json", "r")
user_ids = json.dumps(json.load(users_fd))
users_fd.close()

reviews_fd = open("./yelp_academic_dataset_review.json", "r")

for line in reviews_fd:
    # print(len(review_ids))
    review = json.loads(line)
    total_reviews += 1

    if review['business_id'] in restaurant_ids:
        our_reviews.append(review)
        review_ids.append(review["review_id"])
    if len(review_ids) > 10000:
        break
reviews_fd.close()


with open("json_data/first_10k_reviews.json", "w") as fd:
    json.dump(our_reviews, fd, indent=2)
fd.close()

with open("json_data/first_10k_review_ids.json", "w") as fd:
    json.dump(review_ids, fd, indent=2)
fd.close()

print(total_reviews)
print(len(review_ids))