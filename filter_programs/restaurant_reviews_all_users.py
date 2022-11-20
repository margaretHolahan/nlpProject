# Program to find all restaurant reviewers 
import json

def main():
    total_reviews = 0
    restaurant_review_count = 0

    all_users = []

    restaurant_fd =  open("json_data/restaurant_ids.json", "r")
    restaurant_ids = json.dumps(json.load(restaurant_fd))
    restaurant_fd.close()

    reviews_fd = open("./yelp_academic_dataset_review.json", "r")

    # Get all reviews from the top 3 users 
    for line in reviews_fd:
        review = json.loads(line)
        total_reviews += 1

        if review['business_id'] in restaurant_ids:
            all_users.append(review['user_id'])
            restaurant_review_count += 1
        # print(total_reviews)
    reviews_fd.close()

    with open("json_data/all_users.json", "w") as fd:
        json.dump(all_users, fd, indent=2)
    fd.close()

    # 6990280 Total Reviews
    # print(total_reviews)

    # 3632012 Restaurant Reviews
    print(restaurant_review_count)

main()