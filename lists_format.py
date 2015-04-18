
import pandas as pd
import dateutil

#-----------------------------------------------------------------------------------------------------------------------

SAMPLES_SET_URL   = "../data/lists/samples.csv"
RAW_TWEETS_DB_URL = "../data/twitts-text/raw_tweets"
TEXT_TOKENS_URL   = "../data/twitts-text/text_tokens.npz"

columns = ["class", "user_id", "name", "screen_name", "description", "verified", "location", "lat",
           "lon", "country", "created_at", "followers_count", "friends_count",
           "statuses_count", "favourites_count", "listed_count"]

ts_parser = lambda date_str: dateutil.parser.parse(date_str) if pd.notnull(date_str) else None

visual_class_color = { 0:'r', 1:'g' }
