import numpy as np
import pandas as pd
import math
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split




def filterData(df):
    """Remove non-text posts and removed posts. Concatenate title and text to form content.
    """
    df = df[df['Text_Only'] == True]
    df = df[~df['Title'].isin(['[removed]', '[deleted]', '[deleted by user]', np.nan])].reset_index(drop=True)
    df['Text'] = df['Text'].map(lambda x : '' if x in ['[removed]', '[deleted]', '[deleted by user]', np.nan] else x)
    df['Content'] = df['Title'] + ' ' + df['Text']
    df['Content'] = df['Content'].map(lambda x : str(x).lower())
    return df

def numVotes(score, upvote_ratio):
    if upvote_ratio == 0.5:
        return 0
    if upvote_ratio < 0.5:
        return 21
    else:
        upvotes = math.floor(score/(2-1/upvote_ratio))
        downvotes = upvotes - score
    return upvotes+downvotes

def generateFeatures(df):
    df['Num_votes'] = df[['Upvotes', 'Upvote_ratio']].apply(lambda x : numVotes(x['Upvotes'], x['Upvote_ratio']), axis=1)
    return df

def filterInteractions(df):
    return df.query('Num_comments >= 5 | Num_votes > 15').reset_index(drop=True)


def getTextEmbeddings(text, modelName='all-MiniLM-L6-v2', device='cpu'):
    model = SentenceTransformer(modelName)
    return model.encode(text, device=device, show_progress_bar=True, batch_size=128)


def histogram(df, column, subreddit, log=False):
    figure(figsize=(20, 10))
    plt.hist(df[column], bins=50, log=log, edgecolor='black', color='lightseagreen')
    plt.title(f'Distribution of {column} for {subreddit}.')


def ttSplit(X, y):
    return train_test_split(X, y, test_size=0.2)
