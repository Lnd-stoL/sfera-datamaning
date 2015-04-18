
import datetime
import numpy as np
import pandas as pd
import csv
import lists_format
import pylab as pl
import mpl_toolkits.basemap as bm
import matplotlib.colors as colors

#-----------------------------------------------------------------------------------------------------------------------

def count_users_by_class(group):
    count_pos = sum(1 if user_class == 0 else 0 for user_class in group[1]["class"].values)
    return count_pos, (len(group[1]["class"].values) - count_pos)


def count_users(groups):
    """
    Counts number of positive and negative users
    created at each date.

    Returns:
        count_pos -- 1D numpy array with the counts of positive users created at each date
        count_neg -- 1D numpy array with the counts of negative users created at each date
        dts -- a list of date strings, e.g. ['2014-10-11', '2014-10-12', ...]
    """
    dts = [group[0] for group in groups]
    count_pos, count_neg = np.zeros(len(groups)), np.zeros(len(groups))

    i = 0
    for group in groups:
        count_pos[i], count_neg[i] = count_users_by_class(group)
        i += 1

    return count_pos, count_neg, dts


#-----------------------------------------------------------------------------------------------------------------------
# Read the data

print "Reading data set from " + lists_format.SAMPLES_SET_URL
df_full = pd.read_csv(lists_format.SAMPLES_SET_URL, quoting=csv.QUOTE_NONNUMERIC,
                       sep="\t", header=0, parse_dates=["created_at"], encoding='utf-8', names=lists_format.columns)
print "There are " + str(len(df_full)) + " users in set"

#-----------------------------------------------------------------------------------------------------------------------
# Window initialization

print "Plotting ..."

pl.figure(figsize=(20,20))
window_grid = (10, 10)

fig = pl.gcf()
fig.canvas.set_window_title("Twitter users classification: Exploratory Data Analysis")

#-----------------------------------------------------------------------------------------------------------------------
# Graph #1

# Compute the distribution of the target variable
counts, bins = np.histogram(df_full["class"], bins=[0,1,2])

# Plot the distribution
pl.subplot2grid(window_grid, (7, 0), colspan=2, rowspan=3)
pl.title("Target variable distribution")

barlist = pl.bar(bins[:-1], counts, width=0.6, alpha=0.5)
barlist[0].set_color('r')
barlist[1].set_color('g')

pl.xticks(bins[:-1] + 0.3, ["positive", "negative"])
pl.xlim(bins[0] - 0.5, bins[-1])
pl.ylabel("Number of users")

#-----------------------------------------------------------------------------------------------------------------------
# Graph #2

pl.subplot2grid(window_grid, (7, 2), colspan=8, rowspan=3)
pl.title("Class distribution by account creation month")

grouped = df_full.groupby(map(( lambda dt: dt.strftime("%Y-%m") if pd.notnull(dt) else "NA" ), df_full["created_at"]))
count_pos, count_neg, dts = count_users(grouped)

fraction_pos = count_pos / (count_pos + count_neg + 1e-10)
fraction_neg = 1 - fraction_pos
sort_ind = np.argsort(dts)

pl.bar(np.arange(len(dts)), fraction_pos[sort_ind], width=1.0, color='red', alpha=0.5, linewidth=0, label="Positive")
pl.bar(np.arange(len(dts)), fraction_neg[sort_ind], bottom=fraction_pos[sort_ind], width=1.0, color='green', alpha=0.5,
       linewidth=0, label="Negative")
pl.xticks(np.arange(len(dts)) + 0.4, sorted(dts), rotation=90)
pl.xlim(0, len(dts))
#pl.legend()

#-----------------------------------------------------------------------------------------------------------------------

def plot_points_on_map(map_plot, df_full):
    location_groups = df_full.groupby(["lat", "lon"])
    mlats, mlongs, msizes, mcolors = [], [], [], []

    for group in location_groups:
        if int(group[0][0]) < 0: continue

        pos_in_group, neg_in_group = count_users_by_class(group)
        pos_fraction = float(pos_in_group) / (pos_in_group + neg_in_group)
        neg_fraction = 1 - pos_fraction;

        rgb = (pos_fraction, neg_fraction, 0)
        color = colors.rgb2hex(rgb)

        mlats.append(group[0][0])
        mlongs.append(group[0][1])
        marker_size = (pos_in_group + neg_in_group) * 30
        if marker_size > 200 : marker_size = 200
        msizes.append(marker_size)
        mcolors.append(color.replace('-', '0')[0:7])

    map_plot.scatter(mlongs, mlats, msizes, mcolors, marker='o', zorder=10,
                     linewidths=np.zeros(len(mlongs)).tolist(), alpha=0.75)

#-----------------------------------------------------------------------------------------------------------------------
# Graph #3

pl.subplot2grid(window_grid, (0, 0), colspan=10, rowspan=7)
pl.title("Geospatial distribution of twitter users")

map_plot = bm.Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
map_plot.drawcountries(linewidth=0.4)
map_plot.fillcontinents(color='lavender', lake_color='#000040')
map_plot.drawmapboundary(linewidth=0.4, fill_color='#000040')
map_plot.drawparallels(np.arange(-90,90,30),labels=[0,0,0,0], color='white', linewidth=0.5)
map_plot.drawmeridians(np.arange(0,360,30),labels=[0,0,0,0], color='white', linewidth=0.5)

plot_points_on_map(map_plot, df_full)

points, labels = [], []
points.append(pl.scatter([], [], s=200, edgecolors='none', color='r'))
points.append(pl.scatter([], [], s=200, edgecolors='none', color='g'))
points.append(pl.scatter([], [], s=100, edgecolors='none', color='b'))
points.append(pl.scatter([], [], s=250, edgecolors='none', color='#888800'))
labels.append("Positive")
labels.append("Negative")
labels.append("Circle size   = users amount")
labels.append("Circle color = classes proportion")

pl.legend(points, labels, title='User classes legend',
          ncol=2, frameon=True, fontsize=10,
          loc=4, borderpad=1.8, handletextpad=1,
          scatterpoints=1, labelspacing=1)

#-----------------------------------------------------------------------------------------------------------------------
# Finally show the window

pl.tight_layout()
pl.subplots_adjust(top=0.97, hspace=0.4)
pl.show()
