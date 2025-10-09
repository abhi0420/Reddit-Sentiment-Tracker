import pandas as pd
import matplotlib.pyplot as plt

def _safe_str(value):
    """Safely convert any value to string, handling PRAW objects"""
    if value is None:
        return ""
    elif hasattr(value, 'name'):  # PRAW Redditor object
        return str(value.name)
    elif hasattr(value, '__str__'):
        return str(value)
    else:
        return ""

def summarize_df(reddit_df : pd.DataFrame) -> dict : 
    summary = {}
    if reddit_df.empty:
        return summary

    avg_sentiment = reddit_df["sentiment_score"].mean()
    summary["average_sentiment"] = float(avg_sentiment) 

    reddit_df["created_on"] = pd.to_datetime(reddit_df["created_on"], unit='s', utc=True)
    trend = (reddit_df.set_index("created_on").sort_index().resample("D")["sentiment_score"].mean().dropna())

    # Fix: Convert pandas Series to dictionary with string dates
    summary["trend"] = {date.strftime('%Y-%m-%d'): float(value) for date, value in trend.items()}

    top_post = reddit_df.loc[reddit_df["score"].idxmax()]
    summary["top_post"] = {
        "title": _safe_str(top_post["title"]),
        "author": _safe_str(top_post["author"]),
        "score": float(top_post["score"]),
        "url": _safe_str(top_post["url"])
    }
    
    # Fix: Change field names to match Pydantic model expectations
    most_neg_post = reddit_df.loc[reddit_df["sentiment_score"].idxmin()]
    summary["most_negative_post"] = {
        "title": _safe_str(most_neg_post["title"]),
        "author": _safe_str(most_neg_post["author"]),
        "score": float(most_neg_post["score"]),
        "url": _safe_str(most_neg_post["url"])
    }

    most_pos_post = reddit_df.loc[reddit_df["sentiment_score"].idxmax()]
    summary["most_positive_post"] = {
        "title": _safe_str(most_pos_post["title"]),
        "author": _safe_str(most_pos_post["author"]),
        "score": float(most_pos_post["score"]),
        "url": _safe_str(most_pos_post["url"])
    }

    return summary


def plot_trend(trend : pd.Series):
    title = "Sentiment over time"
    plt.ylabel("Sentiment Score")
    plt.xlabel("Date")
    trend.plot(marker="o", title="Average Sentiment (hourly)")
    plt.show()