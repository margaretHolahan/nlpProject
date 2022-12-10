## Restaurant Review Star Prediction for Yelp Dataset 
In this repository, we provide sentiment analysis on yelp restaurant reviews to predict star ratings(1-5 stars). We investigate the data set provided by Yelp Dataset. Eight machine learning models are used: Naive Bayes, Smoothing Naive Bayes, Gaussian Naive Bayes, Logistic Regression, KNN, FFNN, LSTM, DistilBERT. After analyzing the performance of each model, the best model for predicting the ratings from reviews is DistilBERT.  

## Dataset

|   | stars  | text |
| :------------ |:---------------:| -----:|
| 0  | 1| We have always enjoyed this restaurant and the... |
| 1  | 2      | Not terrible, just not good. Steak was pretty...|
| 2 |  4  |   I've often passed this swanky little place in ... |
| 3 | 1|   I just got back from Chucks and had to login t... |
| 4 | 5 |   There are few truly great lunch places in Old ... |


## STATS:
6,990,280 Total Reviews from Yelp Dataset \
3,632,012 Restaurant Reviews Found

### FIRST 10K
Found first 2000 reviews for each star for a total of 10k reviews

### TOP 3 REVIEW COUNT
**User 1** \
Total: 3048 Reviews \
1 Star: 68 | 2 Star: 265 | 3 Star: 821 | 4 Star: 1443 | 5 Star: 451 

**User 2** \
Total: 1682 Reviews \
1 Star: 25 | 2 Star: 158 | 3 Star: 500 | 4 Star: 692 | 5 Star: 307 

**User 3** \
Total: 1447 Reviews \
1 Star: 5 | 2 Star: 17 | 3 Star: 318 | 4 Star: 817 | 5 Star: 290 

## FILTER_PROGRAMS
**business_filter.py** \
Program to find all restaurant reviews 

**find_top_3.py** \
Program to find top 3 users and sort their reviews by stars

**restaurant_reviews_all_users.py** \
Program to find all restaurant reviewers 

**review_filter_10k.py** \
Program to find 10k reviews total \
Found first 2000 reviews for each star 

## JSON_DATA
**all_users.json** \
Created in restaurant_reviews_all_users.py file \
List of all users who wrote restaurant reviews 

**restaurant_ids.json** \
Created in business_filter.py file \
List of all restaurant reviews 

**restaurants.json** \
Created in business_filter.py file \
List of all restaurant ids 

### FIRST 10K 
Created in review_filter_10k.py 

**TEST** \
20% of each restaurant review sorted into five files by star (each 400 reviews) 

**TRAIN** \
80% of each restaurant review sorted into five files by star (each 1600 reviews) 

### TOP 3 
Created in find_top_3.py \
**top3_reviewer_ids.json** \
List of top 3 reviewers by reviewer id 

**TOP_3_USER_1** \
List of all reviews by this user sorted into five files by star 

**TOP_3_USER_2** \
List of all reviews by this user sorted into five files by star 

**TOP_3_USER_3** \
List of all reviews by this user sorted into five files by star 
