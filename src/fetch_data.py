import praw
import pandas as pd
import os
from dotenv import load_dotenv  
from functools import lru_cache

load_dotenv()

@lru_cache(maxsize=32)
def get_reddit_instance():
    return praw.Reddit(client_id=os.getenv("REDDIT_CLIENT_ID"),
                       client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                       user_agent=os.getenv("REDDIT_USER_AGENT"),
                       username=os.getenv("REDDIT_USERNAME"),
                       password=os.getenv("REDDIT_PASSWORD"))   


def fetch_reddit_data(search_term, subreddit = "",limit= 100):

    reddit = get_reddit_instance()
    if subreddit == "":
        subreddit = "all"
    try:
        submissions = reddit.subreddit(subreddit).search(search_term, limit=limit)
    except Exception as e:
        print(f"Error fetching submissions: {e}")
        return pd.DataFrame()  
    data = []
    for submission in submissions:
        data.append({"id": submission.id,
                     "created_on": submission.created_utc,
                     "term": search_term,
                     "subreddit": submission.subreddit,
                     "author": submission.author,
                     "title": submission.title,
                     "text": submission.selftext,
                     "score": submission.score,
                     "url": submission.url,
                     "is_self": submission.is_self,
                     "permalink": f"https://www.reddit.com{submission.permalink}"})
    return pd.DataFrame(data)


def main():
    df = fetch_reddit_data("Python", limit=10)
    print(df)

if __name__ == "__main__":
    main()