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

# Find most common element in an array
def find_most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def get_user_ids():
    users = []

    # Get list of all users from all_users.json file
    user_fd =  open("json_data/all_users.json", "r")
    user_ids = json.dumps(json.load(user_fd))
    user_fd.close()
    user_ids = user_ids.split()

    # Remove quotes, and square brackets from the strings
    for i in range(len(user_ids)):
        user_ids[i] = user_ids[i].replace('"', '')
        user_ids[i] = user_ids[i].replace('[', '')
        user_ids[i] = user_ids[i].replace(']', '')
        user_ids[i] = user_ids[i].replace(',', '')

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

    users.append(user_1)
    users.append(user_2)
    users.append(user_3)

    print(user_1, user_2, user_3)
    return users

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

def get_reviews(users):
    reviews_fd = open("./yelp_academic_dataset_review.json", "r")

    for line in reviews_fd:
        review = json.loads(line)
        if review['user_id'] == users[0]:
            filter_by_star(review, user_1_reviews)
        if review['user_id'] == users[1]:
            filter_by_star(review, user_2_reviews)
        if review['user_id'] == users[2]:
            filter_by_star(review, user_3_reviews)
        
def main():
    # Get the top 3 users
    users = get_user_ids()
    add_to_files("json_data/top3_reviewer_ids.json", users)

    # Sort through each review to find reviews by top 3 users
    # Then sort by stars
    get_reviews(users)

    # Get count of each review
    # user_1_star_1 68 user_1_star_2 265 user_1_star_3 821 user_1_star_4 1443 user_1_star_5 451
    # user_2_star_1 25 user_2_star_2 158 user_2_star_3 500 user_2_star_4 692 user_2_star_5 307
    # user_3_star_1 5 user_3_star_2 17 user_3_star_3 318 user_3_star_4 817 user_3_star_5 290
    # print("user_1_star_1", len(user_1_reviews["1 star"]), "user_1_star_2", len(user_1_reviews["2 star"]), "user_1_star_3", len(user_1_reviews["3 star"]), "user_1_star_4", len(user_1_reviews["4 star"]), "user_1_star_5", len(user_1_reviews["5 star"]))
    # print("user_2_star_1", len(user_2_reviews["1 star"]), "user_2_star_2", len(user_2_reviews["2 star"]), "user_2_star_3", len(user_2_reviews["3 star"]), "user_2_star_4", len(user_2_reviews["4 star"]), "user_2_star_5", len(user_2_reviews["5 star"]))
    # print("user_3_star_1", len(user_3_reviews["1 star"]), "user_3_star_2", len(user_3_reviews["2 star"]), "user_3_star_3", len(user_3_reviews["3 star"]), "user_3_star_4", len(user_3_reviews["4 star"]), "user_3_star_5", len(user_3_reviews["5 star"]))
    
    # Add reviews to json files
    add_to_files("top3/json_data/top3_user_1/user_1_star_1.json", user_1_reviews["1 star"])
    add_to_files("top3/json_data/top3_user_1/user_1_star_2.json", user_1_reviews["2 star"])
    add_to_files("top3/json_data/top3_user_1/user_1_star_3.json", user_1_reviews["3 star"])
    add_to_files("top3/json_data/top3_user_1/user_1_star_4.json", user_1_reviews["4 star"])
    add_to_files("json_data/top3_user_1/user_1_star_5.json", user_1_reviews["5 star"])
    
    add_to_files("top3/json_data/top3_user_2/user_2_star_1.json", user_2_reviews["1 star"])
    add_to_files("top3/json_data/top3_user_2/user_2_star_2.json", user_2_reviews["2 star"])
    add_to_files("top3/json_data/top3_user_2/user_2_star_3.json", user_2_reviews["3 star"])
    add_to_files("top3/json_data/top3_user_2/user_2_star_4.json", user_2_reviews["4 star"])
    add_to_files("top3/json_data/top3_user_2/user_2_star_5.json", user_2_reviews["5 star"])

    add_to_files("top3/json_data/top3_user_3/user_3_star_1.json", user_3_reviews["1 star"])
    add_to_files("top3/json_data/top3_user_3/user_3_star_2.json", user_3_reviews["2 star"])
    add_to_files("top3/json_data/top3_user_3/user_3_star_3.json", user_3_reviews["3 star"])
    add_to_files("top3/json_data/top3_user_3/user_3_star_4.json", user_3_reviews["4 star"])
    add_to_files("top3/json_data/top3_user_3/user_3_star_5.json", user_3_reviews["5 star"])

main()