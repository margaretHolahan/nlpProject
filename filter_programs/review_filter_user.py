import json

total_reviews = 0
our_reviews = []
review_ids = []

restaurant_fd =  open("json_data/restaurant_ids.json", "r")
restaurant_ids = json.dumps(json.load(restaurant_fd))
restaurant_fd.close()

users_fd =  open("json_data/top3_user_ids.json", "r")
user_ids = json.dumps(json.load(users_fd))
user_ids = user_ids.split()

for i in range(len(user_ids)):
    user_ids[i] = user_ids[i].replace('"', '')
    user_ids[i] = user_ids[i].replace('[', '')
    user_ids[i] = user_ids[i].replace(']', '')
    user_ids[i] = user_ids[i].replace(',', '')

reviews_fd = open("./yelp_academic_dataset_review.json", "r")
for user in user_ids:
    for line in reviews_fd:
        review = json.loads(line)
        total_reviews += 1
        if review['business_id'] in restaurant_ids and review['user_id'] == user:
            print(review)
            our_reviews.append(review)
            review_ids.append(review["review_id"])
            # print(review)
        if len(review_ids) > 10000:
            break

    user_reviews_file = "json_data/user_reviews_" + str(user) + ".json"
    user_reviews_ids = "json_data/top3_user_ids" + str(user) + ".json"
    with open(user_reviews_file, "w") as fd:
        json.dump(our_reviews, fd, indent=2)
    fd.close()

    with open(user_reviews_ids, "w") as fd:
        json.dump(review_ids, fd, indent=2)
    fd.close()
reviews_fd.close()




print(total_reviews)
print(len(review_ids))