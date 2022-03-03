import praw
import pandas as pd
import math
from praw.models import MoreComments
import os

CLIENT_ID = "bgqec0SNqCEeVnQF21-IIw"
CLIENT_SECRET = "fBz7RUMgJCgaSKYfVDC9GvjA9PKTDg"
USER_AGENT = "jackye666"
#SUBREDDITS = ["leagueoflegends","uofm"]
SUBREDDITS = ["uofm"]
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
)
post_dict = {"Title":[],"Text":[],"Upvotes":[],"Downvotes":[],"Subreddit":[],"Source":[]}
post_num = 5000

def cal_votes(post):
    if post.upvote_ratio == 0 or post.upvote_ratio == 0.5:
        upvotes = 0
        downvotes = 0
    else:
        upvotes = math.floor(post.score/(2-1/post.upvote_ratio))
        downvotes = upvotes - post.score
    return upvotes,downvotes

def traverse_section(subreddit_section, subreddit_name, source):
    for post in subreddit_section:
        upvotes,downvotes = cal_votes(post)
        post_dict["Title"].append(post.title)
        post_dict["Downvotes"].append(downvotes)
        post_dict["Text"].append(post.selftext)
        post_dict["Upvotes"].append(upvotes)
        post_dict["Subreddit"].append(subreddit_name)
        post_dict["Source"].append(source)
def main():
    for SUBREDDIT in SUBREDDITS:
        subreddit = reddit.subreddit(SUBREDDIT)
        print("Display Name:", subreddit.display_name)
        print("Title:", subreddit.title)
        traverse_section(subreddit.top(limit=post_num), SUBREDDIT, "top")
        traverse_section(subreddit.hot(limit=post_num), SUBREDDIT, "hot")
        traverse_section(subreddit.new(limit=post_num), SUBREDDIT, "new")

    posts_df = pd.DataFrame(post_dict)
    posts_df.to_csv("post.csv",mode="w+")
    #print(posts_df)

    

if __name__ == "__main__":
    main()