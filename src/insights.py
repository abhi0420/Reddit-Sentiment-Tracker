import pandas as pd
import matplotlib.pyplot as plt

def summarize_df(reddit_df : pd.DataFrame) -> dict : 
    summary = {}
    if reddit_df.empty:
        return summary

    avg_sentiment = reddit_df["sentiment_score"].mean()
    summary["average_sentiment"] = float(avg_sentiment) 

    reddit_df["created_on"] = pd.to_datetime(reddit_df["created_on"], unit='s', utc=True)
    trend = (reddit_df.set_index("created_on").sort_index().resample("D")["sentiment_score"].mean().dropna())

    summary["trend"] = trend  

    top_post = reddit_df.loc[reddit_df["score"].idxmax()]
    summary["top_post"] = {
        "title": top_post["title"],
        "author": top_post["author"],
        "score": float(top_post["score"]),
        "url": top_post["url"]
    }
    # Score -> no of likes
    # Sentiment Score -> Sentiment of the post
    most_neg_post = reddit_df.loc[reddit_df["sentiment_score"].idxmin()]
    summary["most_neg_post"] = {
        "title": most_neg_post["title"],
        "author": most_neg_post["author"],
        "score": float(most_neg_post["score"]),
        "url": most_neg_post["url"]
    }

    most_pos_post = reddit_df.loc[reddit_df["sentiment_score"].idxmax()]
    summary["most_pos_post"] = {
        "title": most_pos_post["title"],
        "author": most_pos_post["author"],
        "score": float(most_pos_post["score"]),
        "url": most_pos_post["url"]
    }

    return summary


def plot_trend(trend : pd.Series):
    title = "Sentiment over time"
    plt.ylabel("Sentiment Score")
    plt.xlabel("Date")
    trend.plot(marker="o", title="Average Sentiment (hourly)")
    plt.show()