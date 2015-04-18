
# loads local twitts database collected by collect_text_data.py
# transforms raw twitts text to tokens and then to matrix of features per user id
# saves this data

#-----------------------------------------------------------------------------------------------------------------------
# some tunable parameters

MAX_WORD_LENGTH  = 12
MIN_WORD_LENGTH  = 4

TOKENS_DICTIONARY_FILE = '../data/twitts-text/tokens_dictionary.txt'    # here will be dumped a set of words (tokens)
                                                                        # collected from all the twitts
                                                                        # useful to test the tokenization

#-----------------------------------------------------------------------------------------------------------------------
# imports

import numpy as np
import pandas as pd
import re
import shelve
import codecs

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from sklearn.feature_extraction import DictVectorizer

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import lists_format

#-----------------------------------------------------------------------------------------------------------------------
# aux functions

# these are defined here to be used in subsequent functions definitions
# globals are declared here to increase performance

en_stop_words = stopwords.words('english')                             # nearly all our training set are english speakers
en_stop_words += ["it's", "don't", "can", "can't", "will", "could"]    # some more stop words

wordnet_lemmatizer = WordNetLemmatizer()
word_extracting_re = re.compile(r"[\w][\w\'`]*\w", flags=re.UNICODE)   # '-' was specially removed from here


def get_words(text):
    words = [word.lower().replace('_', '') for word in word_extracting_re.findall(text)]
    return words


def _get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'a'
    else:
        return 'n'


def get_tokens(words):
    words = zip(words, [_get_wordnet_pos(pos) for w, pos in pos_tag(words)])
    tokens = [wordnet_lemmatizer.lemmatize(word, pos=w_pos) for word, w_pos in words

              if (MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH) and
              (not word in en_stop_words) and
              (not any(i.isdigit() for i in word))]
    return tokens


def get_tweet_tokens(tweet):
    return get_tokens(get_words(tweet))


def collect_users_tokens(raw_twitts):
    ret_user_ids = []
    ret_user_dicts = []

    counter = 0
    raw_twitts_len = len(raw_twitts.keys())
    for user_id, user_tweets in raw_twitts.iteritems():
        counter += 1
        print counter, "/", raw_twitts_len, ": Collecting tokens dict for ", user_id
        u_dict = {}

        for tweet in user_tweets:
            for tweet_token in get_tweet_tokens(tweet['text']):
                try:
                    u_dict[tweet_token] += 1
                except KeyError:
                    u_dict[tweet_token] = 1

        if len(u_dict.keys()) != 0:
            ret_user_ids.append(user_id)
            ret_user_dicts.append(u_dict)

    return ret_user_ids, ret_user_dicts

#-----------------------------------------------------------------------------------------------------------------------
# Main code

print "Twitter users texts processor"

print "Loading raw tweets from", lists_format.RAW_TWEETS_DB_URL
raw_twitts = shelve.open(lists_format.RAW_TWEETS_DB_URL)

print "Collecting tokens ..."
users, users_tokens = collect_users_tokens(raw_twitts)
raw_twitts.close()

print "Vectorizing ..."
v = DictVectorizer(dtype=np.uint32)
vs = v.fit_transform(users_tokens)
feature_names = v.get_feature_names()
print "Dictionary contains", len(feature_names), "tokens"

print "Saving"
tokens_file = codecs.open(TOKENS_DICTIONARY_FILE, 'w', 'utf-8')
tokens_file_text = unicode('\n').join(feature_names)
tokens_file.write(tokens_file_text)
tokens_file.close()

np.savez(lists_format.TEXT_TOKENS_URL, data_data=vs.data, data_indices=vs.indices, data_indptr=vs.indptr, data_shape=vs.shape,
         users=users, users_tokens=users_tokens, feature_names=feature_names)
