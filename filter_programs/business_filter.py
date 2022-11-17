import json

business_count = 0

restaurants = []
restaurant_ids = []
business_fd = open("./yelp_academic_dataset_business.json", "r")
for line in business_fd:
    business = json.loads(line)
    business_count += 1

    if business['attributes'] == None:
        continue

    if 'RestaurantsTakeOut' in business['attributes'] and business['is_open'] == 1:
        if business['attributes']['RestaurantsTakeOut'] == "True":
            restaurants.append(business)
            restaurant_ids.append(business['business_id'])

business_fd.close()

with open("json_data/restaurants.json", "w") as fd:
    json.dump(restaurants, fd, indent=2)
fd.close()

with open("json_data/restaurant_ids.json", "w") as fd:
    json.dump(restaurant_ids, fd, indent=2)
fd.close()

print("Number of businesses:", business_count)
print("Number of restaurants:", len(restaurant_ids))