import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def cleanDataConcatenate(df):
    df['Text'] = df['Text'].map(lambda text : "" if text in ['[removed]', '[deleted]', '[deleted by user]', np.nan] else text)
    df = df[~df['Title'].isin(['[removed]', '[deleted]', '[deleted by user]', np.nan])].reset_index(drop=True)
    df['Text'] = df['Title'] + ' ' + df['Text']
    df['Text'] = df['Text'].map(lambda text : text.lower())
    return df

def upvoteHist(df):
    subreddit = df['Subreddit'][0]
    figure(figsize=(20, 10))
    plt.hist(df['Upvotes'], bins=np.logspace(np.log10(1.0), np.log10(100000.0), 50), log=True, edgecolor='black', color='green')
    plt.gca().set_xscale("log")
    plt.title(f"Distribution of upvotes for {subreddit}")
    plt.show()

def numWordsHist(df):
    subreddit = df['Subreddit'][0]
    figure(figsize=(20, 10))
    plt.hist(df['Title'].map(lambda text : len(text.split(' '))), bins=50, log=False, edgecolor='black', color='green')
    plt.title(f"Distribution of number of words per post for {subreddit}")
    plt.show()

