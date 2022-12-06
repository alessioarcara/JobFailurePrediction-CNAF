import pandas as pd
import cudf
from matplotlib import pyplot as plt

import nltk
STOPWORDS = nltk.corpus.stopwords.words('english')

def plot_multiple_subplots(data, mainPlotCallback, nrows, ncols, figsize=None, plotTitle=None, remainingPlotsCallbacks=[]):
    """
    Parameters
    ----------
    data : pandas DataFrame
    mainPlotCallback : main plot function to draw multiple plots. 
    remainingPlotsCallbacks : list of callbacks that are used to draw on remaining plots.
    """
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    flattened_axs = axs.ravel()
    plt.subplots_adjust(hspace=0.45)
    
    c = 0
    for key in data:
        mainPlotCallback(data, key, flattened_axs[c])
        c += 1
    for callback in remainingPlotsCallbacks:
        callback(flattened_axs[c])
        c += 1
        
    plt.suptitle(plotTitle, fontsize=16, y=1)
    plt.tight_layout()
    plt.show()
    
def bin_df(df, lower_bound, upper_bound):
    return df.groupby(pd.cut(df.index, bins = [0, lower_bound, upper_bound, len(df)], right=False)).sum()

def preprocess_text(raw_text , filters=None , stopwords=STOPWORDS):
    """
        * filter punctuation
        * to_lower
        * remove stop words (from nltk corpus)
        * remove multiple spaces with one
        * remove leading spaces    
    """
    # filter punctuation and case conversion
    translation_table = {ord(char): ord(' ') for char in filters}
    raw_text = raw_text.str.translate(translation_table)
    raw_text = raw_text.str.lower()
    # remove words whose word length is less than 2 characters
    raw_text = raw_text.str.findall('\w{4,}').str.join(' ')
    # remove stopwords
    stopwords_gpu = cudf.Series(stopwords)
    raw_text = raw_text.str.replace_tokens(stopwords_gpu, ' ')
    # replace multiple spaces with single one and strip leading/trailing spaces
    raw_text = raw_text.str.normalize_spaces( )
    raw_text = raw_text.str.strip(' ')
    return raw_text