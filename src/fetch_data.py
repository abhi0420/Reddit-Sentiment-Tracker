import praw
import pandas as pd
import os

def fetch_reddit_data(search_term, subreddit = "",limit= 100):
    reddit = praw.Reddit(client_id=os.getenv("REDDIT_CLIENT_ID"),
                         client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                         user_agent=os.getenv("REDDIT_USER_AGENT"))
    submissions = reddit.subreddit(subreddit).search(search_term, limit=limit)
    data = []
    for submission in submissions:
        data.append({"title": submission.title, 
                     "score": submission.score})
    return pd.DataFrame(data)
