STATS:
    6990280 Total Reviews from Yelp Dataset
    3632012 Restaurant Reviews Found

    FIRST 10K
        Found first 2000 reviews for each star for a total of 10k reviews

    TOP 3 REVIEW COUNT
        user_1_star_1 68 user_1_star_2 265 user_1_star_3 821 user_1_star_4 1443 user_1_star_5 451
        user_2_star_1 25 user_2_star_2 158 user_2_star_3 500 user_2_star_4 692 user_2_star_5 307
        user_3_star_1 5 user_3_star_2 17 user_3_star_3 318 user_3_star_4 817 user_3_star_5 290

FILTER_PROGRAMS
business_filter.py
    Program to find all restaurant reviews

find_top_3.py
    Program to find top 3 users and sort their reviews by stars

restaurant_reviews_all_users.py
    Program to find all restaurant reviewers 

review_filter_10k.py
    Program to find 10k reviews total
    Found first 2000 reviews for each star

JSON_DATA
all_users.json
    Created in restaurant_reviews_all_users.py file
    List of all users who wrote restaurant reviews 

restaurant_ids.json
    Created in business_filter.py file
    List of all restaurant reviews

restaurants.json
    Created in business_filter.py file
    List of all restaurant ids 

    FIRST 10K
        Created in review_filter_10k.py
        TEST
            20% of each restaurant review sorted into five files by star (each 400 reviews)
        TRAIN
            80% of each restaurant review sorted into five files by star (each 1600 reviews)
    
    TOP 3
        Created in find_top_3.py
        top3_reviewer_ids.json
            List of top 3 reviewers by reviewer id
        TOP_3_USER_1
            List of all reviews by this user sorted into five files by star
        TOP_3_USER_2
            List of all reviews by this user sorted into five files by star
        TOP_3_USER_3
            List of all reviews by this user sorted into five files by star