
# twitter users twitts collector
# connects to twitter.com loads and stores into local db twits for users mentioned in training set
# makes simple text processing (removing web links)

#-----------------------------------------------------------------------------------------------------------------------
# twitter access tokens

CONSUMER_KEY    = "0rEB7f7VriHnl2IkXVi6ljr4R"
CONSUMER_SECRET = "81gzIDWSFFWi0lcyQywIQEASnDbRW6nKTu7W7jIq9SAXeHrSNM"

ACCESS_TOKEN_KEY    = "1239605168-o0bbxvdtugtJ8qkpwEpRAuAuOrUZ766h54lTRbv"
ACCESS_TOKEN_SECRET = "mSxw3S7DDSlC6hx22OkvXoQ9FY3bmFxtmn6YbNAYmIADM"

#-----------------------------------------------------------------------------------------------------------------------
# other tunables

TRAINING_SET_URL = "../data/lists/twitter_train.txt"

BAD_REQUESTS_FILE = "../data/twitts-text/bad_requests.txt"    # request failure id list
OWERWRITE_EXISTING_DB = True
DB_FLUSH_RATE = 50                                            # requests number

#-----------------------------------------------------------------------------------------------------------------------

import twitter
import pandas as pd
import shelve
import re
import time
import signal

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import lists_format

#-----------------------------------------------------------------------------------------------------------------------
# auxilary functions

def get_user_tweets(user_id, twitter_api, weblinks_re):

    user_statuses = None
    while user_statuses is None:
        try:
            user_statuses = twitter_api.GetUserTimeline(str(int(user_id)), exclude_replies=True, trim_user=True,
                                                        count=200, include_rts=False)
        except twitter.TwitterError as ex:
            print "Twitter API Error:", ex
            try:
                if ex.message[0]['code'] == 88:    # rate limit exceeded
                    sleep_time = int(twitter_api.GetSleepTime("statuses/user_timeline")) + 1
                    print "Twitter API: Query limit exceeded. Sleeping for", sleep_time, "seconds (", float(sleep_time)/60, " minutes) ..."
                    time.sleep(sleep_time)
                else:
                    return None
            except Exception as exc:
                print exc
                return None

    statuses_dicts = [status.AsDict() for status in user_statuses]
    statuses_dicts = [status_dict for status_dict in statuses_dicts if status_dict["retweeted"] == False]
    for status in statuses_dicts: status['text'] = weblinks_re.sub('', status['text'])   # to filter links in media posts

    return statuses_dicts

#-----------------------------------------------------------------------------------------------------------------------
# Main code

print "Twitter users texts collector"

print "Initializing Twitter API ..."
twitter_api = twitter.Api(consumer_key=CONSUMER_KEY,
                  consumer_secret=CONSUMER_SECRET,
                  access_token_key=ACCESS_TOKEN_KEY,
                  access_token_secret=ACCESS_TOKEN_SECRET)

print "Loading training set from", TRAINING_SET_URL
df_train_users = pd.read_csv(TRAINING_SET_URL, sep=",", header=0, names=["user_id", "class"])

out_data = shelve.open(lists_format.RAW_TWEETS_DB_URL)

#-----------------------------------------------------------------------------------------------------------------------
# Signal handling

# used to stop the collection process and save database correctly
def sigint_handler(signum, frame):
    print 'Terminating due to', signum
    out_data.close()
    working = False

signal.signal(signal.SIGINT, sigint_handler)

#-----------------------------------------------------------------------------------------------------------------------
# Data doanloading

print "Downloading data from twitter"
working = True

counter = 0
users_list = df_train_users["user_id"]
weblinks_re = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
bad_requests = []

for user_id in users_list:
    if not working: break

    counter += 1
    print counter, '/', str(len(users_list)) + ':', "Getting tweets for user_id =", user_id

    if OWERWRITE_EXISTING_DB == False and out_data.has_key(str(user_id)):
        print "Already in database"
    else:
        user_texts = get_user_tweets(user_id, twitter_api, weblinks_re)
        if not user_texts is None:
            print 'Fetched', len(user_texts), "tweets - ok"
            out_data[str(user_id)] = user_texts
        else:
            bad_requests.append(str(user_id))

    if counter % 50 == 0:
        print 'Saving bad requests ...'
        bad_dump_file = open(BAD_REQUESTS_FILE, 'w')
        bad_dump_file.write('\n'.join(bad_requests))
        bad_dump_file.close()

        print "Flushing database to disk"
        out_data.sync()

print "Saving raw tweets database"
out_data.close()
print "Ready"
