# Program to find top 3 users and sort their reviews by stars
import json
from collections import Counter

user_1_reviews = {"1 star" : [], "2 star" : [], "3 star" : [], "4 star" : [], "5 star" : []}
user_2_reviews = {"1 star" : [], "2 star" : [], "3 star" : [], "4 star" : [], "5 star" : []}
user_3_reviews = {"1 star" : [], "2 star" : [], "3 star" : [], "4 star" : [], "5 star" : []}

def add_to_files(file_path, array):
    with open(file_path, "w") as fd:
        json.dump(array, fd, indent=2)
    fd.close()

def filter_resaurants():
    # Store all restaurant reviews
    restaurant_reviews = []
    user_ids = []

    # Get list of all restaurants from restaurants.json file
    restaurant_fd =  open("json_data/restaurant_ids.json", "r")
    restaurant_ids = json.dumps(json.load(restaurant_fd))
    restaurant_fd.close()

    reviews_fd = open("./yelp_academic_dataset_review.json", "r", encoding='utf-8')
    for line in reviews_fd:
        review = json.loads(line)
        if review['business_id'] in restaurant_ids:
            restaurant_reviews.append(review)
            user_ids.append(review['user_id'])
        
    return restaurant_reviews, user_ids 

# Find most common element in an array
def find_most_common(lst):
    data = Counter(lst)
    count = 0
    return data.most_common(1)[0][0]

def get_user_ids(user_ids):
    top_users = []

    # Find the top reviewer with the most reviews
    user_1 = find_most_common(user_ids)
    # Remove all instances of this reviewer from the list of all user_ids per review
    user_ids[:] = (value for value in user_ids if value != user_1)
    
    # Find the second most active reviewer 
    user_2 = find_most_common(user_ids)
    # Remove all instances of this reviewer from the list of all user_ids per review
    user_ids[:] = (value for value in user_ids if value != user_2)

   # Find the third most active reviewer 
    user_3 = find_most_common(user_ids)

    top_users.append(user_1)
    top_users.append(user_2)
    top_users.append(user_3)

    print(user_1, user_2, user_3)
    return top_users

def filter_by_star(cur_review, cur_user):
    if cur_review['stars'] == 1.0:
        cur_user["1 star"].append(cur_review)
    if cur_review['stars'] == 2.0:
        cur_user["2 star"].append(cur_review)
    if cur_review['stars'] == 3.0:
        cur_user["3 star"].append(cur_review)
    if cur_review['stars'] == 4.0:
        cur_user["4 star"].append(cur_review)
    if cur_review['stars'] == 5.0:
        cur_user["5 star"].append(cur_review)

def filter_review_by_user(restaurant_reviews, user_ids):
    for review in restaurant_reviews:
        if review['user_id'] == user_ids[0]:
            filter_by_star(review, user_1_reviews)
        if review['user_id'] == user_ids[1]:
            filter_by_star(review, user_2_reviews)
        if review['user_id'] == user_ids[2]:
            filter_by_star(review, user_3_reviews)
        
def main():
    # Sort through each review to find reviews by top 3 users
    # Then sort by stars
    restaurant_reviews, user_ids = filter_resaurants()
    # print("user_ids", user_ids)
    top_users = get_user_ids(user_ids)

    # Find all reviews for the top 3 users
    filter_review_by_user(restaurant_reviews, top_users)

    # Get count of each review
    # user_1_star_1 68 user_1_star_2 265 user_1_star_3 821 user_1_star_4 1443 user_1_star_5 451
    # user_2_star_1 25 user_2_star_2 158 user_2_star_3 500 user_2_star_4 692 user_2_star_5 307
    # user_3_star_1 5 user_3_star_2 17 user_3_star_3 318 user_3_star_4 817 user_3_star_5 290
    print("user_1_star_1", len(user_1_reviews["1 star"]), "user_1_star_2", len(user_1_reviews["2 star"]), "user_1_star_3", len(user_1_reviews["3 star"]), "user_1_star_4", len(user_1_reviews["4 star"]), "user_1_star_5", len(user_1_reviews["5 star"]))
    print("user_2_star_1", len(user_2_reviews["1 star"]), "user_2_star_2", len(user_2_reviews["2 star"]), "user_2_star_3", len(user_2_reviews["3 star"]), "user_2_star_4", len(user_2_reviews["4 star"]), "user_2_star_5", len(user_2_reviews["5 star"]))
    print("user_3_star_1", len(user_3_reviews["1 star"]), "user_3_star_2", len(user_3_reviews["2 star"]), "user_3_star_3", len(user_3_reviews["3 star"]), "user_3_star_4", len(user_3_reviews["4 star"]), "user_3_star_5", len(user_3_reviews["5 star"]))
    
    # Add reviews to json files
    add_to_files("json_data/top3/top3_user_1/user_1_star_1.json", user_1_reviews["1 star"])
    add_to_files("json_data/top3/top3_user_1/user_1_star_2.json", user_1_reviews["2 star"])
    add_to_files("json_data/top3/top3_user_1/user_1_star_3.json", user_1_reviews["3 star"])
    add_to_files("json_data/top3/top3_user_1/user_1_star_4.json", user_1_reviews["4 star"])
    add_to_files("json_data/top3/top3_user_1/user_1_star_5.json", user_1_reviews["5 star"])
    
    add_to_files("json_data/top3/top3_user_2/user_2_star_1.json", user_2_reviews["1 star"])
    add_to_files("json_data/top3/top3_user_2/user_2_star_2.json", user_2_reviews["2 star"])
    add_to_files("json_data/top3/top3_user_2/user_2_star_3.json", user_2_reviews["3 star"])
    add_to_files("json_data/top3/top3_user_2/user_2_star_4.json", user_2_reviews["4 star"])
    add_to_files("json_data/top3/top3_user_2/user_2_star_5.json", user_2_reviews["5 star"])
    
    add_to_files("json_data/top3/top3_user_3/user_3_star_1.json", user_3_reviews["1 star"])
    add_to_files("json_data/top3/top3_user_3/user_3_star_2.json", user_3_reviews["2 star"])
    add_to_files("json_data/top3/top3_user_3/user_3_star_3.json", user_3_reviews["3 star"])
    add_to_files("json_data/top3/top3_user_3/user_3_star_4.json", user_3_reviews["4 star"])
    add_to_files("json_data/top3/top3_user_3/user_3_star_5.json", user_3_reviews["5 star"])
    
main()