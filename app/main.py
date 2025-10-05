from fetch_data import fetch_reddit_data
from analyze import analyze_sentiment
from insights import summarize_df, plot_trend

def main():
    reddit_df = fetch_reddit_data("Trump", "usa", limit=140)
    print("Reddit Df : ",reddit_df)
    sentiment_results = []
    reddit_df.loc[reddit_df['text'].str.len() == 0, "text"] = reddit_df["title"]
    for row in reddit_df.itertuples():
        text = row.text
        aspect = row.term
        title = row.title
        sentiment_result = analyze_sentiment(text, title, aspect)
        sentiment_results.append(sentiment_result) 
    print("Sentiment Results: ", sentiment_results)
    reddit_df["sentiment"] = [i['label'] for i in sentiment_results]
    reddit_df["sentiment_score"] = [i['score'] for i in sentiment_results]
    print("Reddit Df with Sentiment: ", reddit_df[["text", "term", "sentiment", "sentiment_score"]])

    insights = summarize_df(reddit_df)
    print("Insights: ", insights)

    if "trend" in insights:
        plot_trend(insights["trend"])

if __name__ == "__main__":
    main()