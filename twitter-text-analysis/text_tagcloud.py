
# draws, shows and saves the tag cloud

#-----------------------------------------------------------------------------------------------------------------------
# some tunable parameters

WORD_FREQ_FILE_PATH   = "../data/twitts-text/words_frequency_statistics.txt"

TAGCLOUD_MASK_PATH    = "../data/twitts-text/tag_cloud_mask.png"
TAGCLOUD_IMAGE_PATH   = "../data/twitts-text/tag_cloud.png"
TAGCLOUD_MAXWORDS     = 1100              # how many most frequent words will be drawn
                                          # adjusted to show words with more than 2% frequency (~60 matches)
                                          # adjusted using WORD_FREQ_FILE_PATH

#-----------------------------------------------------------------------------------------------------------------------
# imports

import numpy as np
import codecs

from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from scipy.misc import imread
from scipy.sparse import csr_matrix

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import lists_format

#-----------------------------------------------------------------------------------------------------------------------
# main code

print "Twitter users tag cloud drawer"

print "Loading tokens from", lists_format.TEXT_TOKENS_URL
tokens_data_file = np.load(lists_format.TEXT_TOKENS_URL)
tokens_matrix = csr_matrix((tokens_data_file['data_data'], tokens_data_file['data_indices'],
                            tokens_data_file['data_indptr']), tokens_data_file['data_shape'])
tokens = tokens_data_file['users_tokens']
feature_names = tokens_data_file['feature_names']

print "Calculating words frequency statistic ..."
words = zip(feature_names, tokens_matrix.sum(0).tolist()[0])

print "Sorting words by frequency"
words.sort(key=lambda wf: wf[1], reverse=True)

print "Saving words frequency statistics to", WORD_FREQ_FILE_PATH
word_freq_file = codecs.open(WORD_FREQ_FILE_PATH, 'w', 'utf-8')
max_word_freq = words[0][1]
for word, freq in words:
    word_freq_file.write(unicode(word) + '  ' + unicode(str(int(freq)))
                         + '  ' + unicode(str(int(freq * 100 / max_word_freq)) + '%') + unicode('\n'))
word_freq_file.close()

#-----------------------------------------------------------------------------------------------------------------------
# drawing the cloud

print "Drawing tags cloud ..."

wordcloud_mask = imread(TAGCLOUD_MASK_PATH)
wordcloud = WordCloud(background_color="white", mask=wordcloud_mask)
wordcloud.fit_words(words[:TAGCLOUD_MAXWORDS])
wordcloud.to_file(TAGCLOUD_IMAGE_PATH)

plt.title("Twitter users tags cloud")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout()

plt.show()
