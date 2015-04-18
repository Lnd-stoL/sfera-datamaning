
#-----------------------------------------------------------------------------------------------------------------------
# twitter access settings

CONSUMER_KEY    = "0rEB7f7VriHnl2IkXVi6ljr4R"
CONSUMER_SECRET = "81gzIDWSFFWi0lcyQywIQEASnDbRW6nKTu7W7jIq9SAXeHrSNM"

ACCESS_TOKEN_KEY    = "1239605168-o0bbxvdtugtJ8qkpwEpRAuAuOrUZ766h54lTRbv"
ACCESS_TOKEN_SECRET = "mSxw3S7DDSlC6hx22OkvXoQ9FY3bmFxtmn6YbNAYmIADM"

GEO_USER_NAME = ["lndstol", "f772728", "dudu123456", "menica", "vonaca", "yeca1234", "xitay"]

#-----------------------------------------------------------------------------------------------------------------------
# other tunables

import lists_format
TRAINING_SET_URL = "./lists/twitter_train.txt"

#-----------------------------------------------------------------------------------------------------------------------

import math
import twitter
import pandas as pd
import requests
import time
import csv
import re
import sys

#-----------------------------------------------------------------------------------------------------------------------

def get_coordinates_by_location(location):
    """
    This function gets geographic coordinates and city name
    form external web service GeoNames using 'location' string.
    """

    location_parts = []
    for p1 in location.replace('#', '').replace('~', '').replace(')', '').replace(':', '').split('and'):
        for p2 in p1.split('/'):
            for p3 in p2.split(','):
                for p4 in p3.split('|'):
                    for p5 in p4.split(' '):
                        for p6 in p5.split('&'):
                              location_parts.append(p6.strip())


    for location_part in location_parts:
        if "ngland" in location_part: location_part = "England"
        if not re.match("^[ .A-Za-z0-9_-]*$", location_part): continue
        sys.stdout.write("[" + location_part + "] ")

        for i in range(0, len(GEO_USER_NAME)*2):
            get_coordinates_by_location.last_geo_user_id += 1;
            geo_user_name = GEO_USER_NAME[get_coordinates_by_location.last_geo_user_id % len(GEO_USER_NAME)]

            query = None
            try:
                query = requests.get('http://api.geonames.org/searchJSON' +
                                     "?q='" + location_part + "'"
                                     '&maxRows=1' +
                                     '&operator=OR' +
                                     '&username=' + geo_user_name)
            except requests.ConnectionError as err:
                print "Error occured while connecting geonames.org: " + err.message()

            responseJSON = query.json();
            if "status" in responseJSON:
                if (responseJSON["status"]["value"] in (18, 19, 20)):      # hour, week or day limits exceeded
                    #sleep_time = 1;
                    print(geo_user_name + ": Query limit exceeded.")
                    #time.sleep(sleep_time);
                    continue
                else:
                    print("Error getting data from GeoName for " + location + ": " + responseJSON["status"]["message"])
                    return (-1, -1, "")

            if "geonames" in responseJSON and len(responseJSON["geonames"]) != 0:
                found_location = responseJSON["geonames"][0]
                return (found_location["lat"], found_location["lng"], found_location["countryName"])

    print "Location lookup failed for '" + location + "'";
    return (-2, -2, "")
get_coordinates_by_location.last_geo_user_id = 0


def twitter_user_to_dataframe_record(user):
    record = {
        "user_id": user.id,
        "name": user.name,
        "screen_name": user.screen_name,
        "created_at": lists_format.ts_parser(user.created_at),
        "followers_count": user.followers_count,
        "friends_count": user.friends_count,
        "statuses_count": user.statuses_count,
        "favourites_count": user.favourites_count,
        "listed_count": user.listed_count,
        "verified": user.verified
    }

    if user.description is not None and user.description.strip() != "":
        record["description"] = user.description

    if user.location is not None and user.location.strip() != "":
        record["location"] = user.location
        record["lat"], record["lon"], record["country"] = get_coordinates_by_location(user.location)

    if "country" in record:
        print(record["country"])
    else:
        print("Failed to detect")

    return record


def get_user_records(df):
    user_records = []
    ids_list = df.icol(0).tolist()

    twitter_api = twitter.Api(consumer_key=CONSUMER_KEY,
                              consumer_secret=CONSUMER_SECRET,
                              access_token_key=ACCESS_TOKEN_KEY,
                              access_token_secret=ACCESS_TOKEN_SECRET)

    one_query_user_count = 100
    loaded_user_count = 0
    for i in range(0, int(math.ceil(float(len(ids_list)) / one_query_user_count))):
        one_query_ids = ids_list[i*one_query_user_count : min((i+1)*one_query_user_count, len(ids_list))]

        query = None
        while True:
            try:
                query = twitter_api.UsersLookup(user_id=one_query_ids)
                break
            except twitter.TwitterError as ex:
                if ex[0][0]["code"] == 88:
                    print "Query limit exceeded. Sleeping for a minute ..."
                    time.sleep(60)
                    continue
                else:
                    print("Warning: " + str(ex.message))
                    break

        if query is None: continue

        for user in query:
            loaded_user_count += 1
            sys.stdout.write(str(loaded_user_count) + ': ' + user.screen_name + " @ ")
            user_record = twitter_user_to_dataframe_record(user)
            user_records.append(user_record)

    return user_records


#-----------------------------------------------------------------------------------------------------------------------
# Main

print("Twitter data collector")

print("Loading training set from " + TRAINING_SET_URL)
df_train_users = pd.read_csv(TRAINING_SET_URL, sep=",", header=0, names=["user_id", "class"])

print("Downloading data from twitter")
user_records = get_user_records(df_train_users)

print "Creating data frame from loaded data"
df_records_columns = ["user_id", "name", "screen_name", "description", "verified", "location", "lat",
                                   "lon", "country", "created_at", "followers_count", "friends_count",
                                   "statuses_count", "favourites_count", "listed_count"]
df_records = pd.DataFrame(user_records, columns=df_records_columns)

print "Merging data frame with the training set"
df_full = pd.merge(df_train_users, df_records, on="user_id", how="left")
print "Finished building data frame"

print "Saving samples to " + lists_format.SAMPLES_SET_URL
df_full.to_csv(lists_format.SAMPLES_SET_URL, sep="\t", encoding='utf-8', columns=lists_format.columns,
               index=False, quoting=csv.QUOTE_NONNUMERIC)
