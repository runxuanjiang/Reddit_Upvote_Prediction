import praw
import pandas as pd
import datetime as dt
from psaw import PushshiftAPI
import sys

now = int(dt.datetime.now().timestamp()) - 300000
reddit = praw.Reddit(
    client_id="b_A8n4V9324jdqlqRIp6Sw",
    client_secret="aMaQ1dwE4maj8iyF8HaXqHTwg34x4Q",
    user_agent="testscript",
)

api = PushshiftAPI(reddit)



def collect_subreddit(subreddit_name):
    start_time = now
    counter = 0
    while True:
        post_dict = {"Title":[], "Text":[], "Timestamp": [], "Upvotes":[], "Upvote_ratio":[], "Num_comments":[], "Nsfw":[], "Text_Only":[], "Subreddit":[], "Url": []}
        empty = True
        for post in api.search_submissions(before=start_time, subreddit=subreddit_name, limit=1000):
            empty = False
            post_dict["Title"].append(post.title)
            post_dict["Text"].append(post.selftext)
            post_dict["Timestamp"].append(post.created_utc)
            post_dict["Upvotes"].append(post.score)
            post_dict['Upvote_ratio'].append(post.upvote_ratio)
            post_dict["Num_comments"].append(post.num_comments)
            post_dict['Nsfw'].append(post.over_18)
            post_dict['Text_Only'].append(post.is_self)
            post_dict["Subreddit"].append(post.subreddit.display_name)
            post_dict['Url'].append(post.url)
            start_time = int(post.created_utc)

        if empty:
            print("Post collection finished")
            break

        pd.DataFrame(post_dict).to_csv(f"{subreddit_name}.csv", mode='a', index=False, header=False)
        counter += 1
        print(f"Posts collected: {1000 * counter}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stream_data.py <subreddit name>")
        quit()
    subreddit_name = sys.argv[1]
    post_dict = {"Title":[], "Text":[], "Timestamp": [], "Upvotes":[], "Upvote_ratio":[], "Num_comments":[], "Nsfw":[], "Text_Only":[], "Subreddit":[], "Url": []}
    pd.DataFrame(post_dict).to_csv(f"{subreddit_name}.csv", index=False)
    collect_subreddit(subreddit_name)