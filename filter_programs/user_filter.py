import json

total_users = 0
our_users = []
user_ids = []

user_fd = open("yelp_academic_dataset_user.json", "r")

for line in user_fd:
    user = json.loads(line)
    total_users += 1

    if 'review_count' in user:
        if user['review_count'] > 15000:
            del user['friends']
            our_users.append(user)
            user_ids.append(user['user_id'])
        if len(user_ids) > 9:
            break
user_fd.close()

with open("json_data/top3_users.json", "w") as fd:
    json.dump(our_users, fd, indent=2)
fd.close()

with open("json_data/top3_user_ids.json", "w") as fd:
    json.dump(user_ids, fd, indent=2)
fd.close()

print(total_users)
print(len(our_users))