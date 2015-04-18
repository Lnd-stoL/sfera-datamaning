
#-----------------------------------------------------------------------------------------------------------------------
# tune plotting

PLOT_GEO_GROUP        = True
PLOT_SOCIAL_GROUP     = True
PLOT_SOCIAL_LOG_GROUP = True
PLOT_OTHERS_GROUP     = True

#-----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import pylab as pl
import sklearn.preprocessing as sp
import csv
import re
import dateutil
import lists_format

#-----------------------------------------------------------------------------------------------------------------------

def create_new_features(df_users, features):
    print "\t<building country codes ...>"
    country_codes = { country:i  for i, country in enumerate(list(set(df_users["country"].tolist()))) }

    feature_calculators = {
        "name_words":          lambda record:  len(re.split('[;:.,\-\d ]+', unicode(record["name"]))),
        "screen_name_length":  lambda record:  len(unicode(record["screen_name"])),
        "description_length":  lambda record:  len(unicode(record["description"])),
        "created_year":        lambda record:  record["created_at"].year,
        "country_code":        lambda record:  country_codes[record["country"]],
        "verified":            lambda record:  int(bool(record["verified"]))
    }

    for feature in feature_calculators.keys():
        calculator = feature_calculators[feature]
        df_users[feature] = [calculator(user) for _, user in df_users.iterrows()]
        print "\tcalculated for: " + feature

    new_features = [feature for feature in df_users.columns.values
                    if feature in features or feature in feature_calculators.keys()]

    return df_users, new_features


def find_correlated_features(x, features):
    treshold = 0.2

    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):
            if i < j:
                cor_coeff = np.corrcoef(x[i], x[j])[0][1]
                if np.abs(cor_coeff) > treshold:
                    print "Correlated features: %s + %s -> %.2f" % (feature_i, feature_j, cor_coeff)

#-----------------------------------------------------------------------------------------------------------------------

np.set_printoptions(linewidth=150, precision=3, suppress=True)

print "Loading collected dataset from '" + lists_format.SAMPLES_SET_URL + "'"
df_users = pd.read_csv(lists_format.SAMPLES_SET_URL, sep="\t", header=0, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC,
                       converters={"created_at": lists_format.ts_parser}, names=lists_format.columns)

# Remove rows with users not found
df_users = df_users[pd.notnull(df_users['name'])]
df_users["lat"].fillna(value=0, inplace=True)
df_users["lon"].fillna(value=0, inplace=True)
df_users["description"].fillna(value="", inplace=True)

print "Calculating new features ..."
features = ["lat", "lon", "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count"]
df_users, features = create_new_features(df_users, features)

x = df_users[features].values
y = df_users["class"].values

print "Analyzing correlation:"
find_correlated_features(x, features)

#-----------------------------------------------------------------------------------------------------------------------

def plot_two_features_scatter(x_i, x_j, clrs):
    pl.scatter(x_j, x_i, s=12, c=clrs, alpha=0.6, linewidths=0)


def plot_feature_histogram(x_i, y):
    users_by_class = { 0:[], 1:[] }
    for usr, ucl in zip(x_i, y):  users_by_class[ucl].append(usr)
    pl.hist([users_by_class[0], users_by_class[1]], bins=15, color=('r', 'g'), alpha=0.7, linewidth=0)


def plot_dataset(x, y, features, name):
    pl.rcParams['toolbar'] = 'None'
    pl.figure(figsize=(14, 14))
    fig = pl.gcf()
    fig.canvas.set_window_title(name)

    user_colors = [lists_format.visual_class_color[ucl] for ucl in y]

    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):

            ax = pl.subplot2grid((len(features), len(features)), (i, j))
            ax.xaxis.set_label_position('top')

            ax.set_xticklabels([])
            ax.set_yticklabels([])

            if i == 0:  pl.xlabel(feature_j)
            if j == 0:  pl.ylabel(feature_i)

            if i != j:
                plot_two_features_scatter(x[:, i], x[:, j], user_colors)
            else:
                plot_feature_histogram(x[:, i], y)

    pl.tight_layout()
    pl.subplots_adjust(hspace=0.03, wspace=0.03)
    pl.show()


def log_transform_features(data, features, transformed_features):
    for transform_feature in transformed_features:
        cind = features.index(transform_feature)
        data[:, cind] = np.log(1 + data[:, cind])
    return data

#-----------------------------------------------------------------------------------------------------------------------

#---- Plotting geographical features correlation
geo_features_new = ["lat", "lon", "country_code"]
geo_features = [f for f in geo_features_new if f in features]
geo_feature_ind = [i for i, f in enumerate(features) if f in geo_features]

if PLOT_GEO_GROUP:
    print "Plotting geographical features group ..."
    plot_dataset(x[:, geo_feature_ind], y, geo_features, "Geo features group")


#---- Plotting social features correlation
social_features_new = ["verified", "followers_count", "friends_count", "statuses_count", "favourites_count",
                       "listed_count", "created_year"]
social_features = [f for f in social_features_new if f in features]
social_feature_ind = [i for i, f in enumerate(features) if f in social_features]

if PLOT_SOCIAL_GROUP:
    print "Plotting social features group ..."
    plot_dataset(x[:, social_feature_ind], y, social_features, "Social features group")

transformed_features = ["followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count"]
x = log_transform_features(x, features, transformed_features)

if PLOT_SOCIAL_LOG_GROUP:
    print "Plotting social (log) features group ..."
    plot_dataset(x[:, social_feature_ind], y, social_features, "Social features group (logarithmical)")

#---- Plotting other features correlation
other_features_new = ["name_words", "screen_name_length", "description_length"]
other_features = [f for f in other_features_new if f in features]
other_feature_ind = [i for i, f in enumerate(features) if f in other_features]

if PLOT_OTHERS_GROUP:
    print "Plotting other features group ..."
    plot_dataset(x[:, other_feature_ind], y, other_features, "Other features group")

#-----------------------------------------------------------------------------------------------------------------------

selected_features = ["followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count", "created_year", "name_words", "screen_name_length", "description_length"]
selected_features_ind = [i for i, f in enumerate(features) if f in selected_features]

x_1 = np.nan_to_num(x[:, selected_features_ind])
x_min = x_1.min(axis=0)
x_max = x_1.max(axis=0)
x_new = (x_1 - x_min) / (x_max - x_min)

df_out = pd.DataFrame(data=x_new, index=df_users["user_id"], columns=selected_features)
df_out.to_csv("../data/lists/features.csv", sep="\t")
