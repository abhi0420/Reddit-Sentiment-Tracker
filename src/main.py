from fetch_data import fetch_reddit_data
from analyze import analyze_sentiment

def main():
    reddit_df = fetch_reddit_data("technology", limit=10)
    print("Reddit Df : ",reddit_df)
    sentiment_results = []
    for row in reddit_df.itertuples():
        text = row.text
        aspect = row.term
        sentiment_result = analyze_sentiment(text, aspect)
        sentiment_results.append({"text": text, "aspect": aspect, "sentiment": sentiment_result})
    print("Sentiment Results: ", sentiment_results)
    reddit_df["sentiment"] = sentiment_results
    print("Reddit Df with Sentiment: ", reddit_df[["text", "term", "sentiment"]])

if __name__ == "__main__":
    main()