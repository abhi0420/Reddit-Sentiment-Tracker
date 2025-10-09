from services.fetch_data import fetch_reddit_data
from services.analyze import analyze_sentiment
from insights import summarize_df, plot_trend

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict, List
from models import AnalyzeRequest, AnalyzeResponse, Post   
 

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


app = FastAPI()

MAX_WORKERS = 8

@app.get("/health")
def health():
    return {"status": "ok"}

def _make_empty_response() -> Dict[str, Any]:
    return {
        "posts": [],
        "summary": {
            "average_sentiment": 0.0,
            "trend": {},
            "top_post": {},
            "most_positive_post": {},
            "most_negative_post": {}
        }

    }

def _safe_record_from_row(row : Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(row)
    if "score" in rec:
        rec["score"] = int(rec.get("score", 0))
    if "num_comments" in rec:
        rec["num_comments"] = int(rec.get("num_comments", 0))

    if "sentiment_score" in rec:
        rec["sentiment_score"] = float(rec.get("sentiment_score", 0.0))

    # Fix: Ensure created_on_utc is always present and properly formatted
    if "created_on_utc" in rec:
        if pd.isna(rec["created_on_utc"]) or rec["created_on_utc"] is None:
            rec["created_on_utc"] = pd.Timestamp.now().isoformat()
        elif not isinstance(rec["created_on_utc"], str):
            rec["created_on_utc"] = pd.to_datetime(rec["created_on_utc"]).isoformat()
    else:
        # Add missing created_on_utc field
        rec["created_on_utc"] = pd.Timestamp.now().isoformat()

    # Fix: Convert sentiment string to numeric if Post model expects a number
    # You'll need to map sentiment labels to numbers
    if "sentiment" in rec and isinstance(rec["sentiment"], str):
        sentiment_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        rec["sentiment"] = sentiment_map.get(rec["sentiment"].lower(), 0.0)

    # Fix: Handle PRAW objects and ensure all fields are strings
    for col in ["title", "text", "term", "subreddit", "author", "url", "permalink"]:
        value = rec.get(col)
        if value is None:
            rec[col] = ""
        elif hasattr(value, 'name'):  # PRAW Redditor object has a 'name' attribute
            rec[col] = str(value.name)
        elif hasattr(value, '__str__'):  # Any other object with string representation
            rec[col] = str(value)
        else:
            rec[col] = ""

    return rec

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):

    try:
        reddit_df = fetch_reddit_data(
            request.text,
            subreddit=request.subreddit,
            limit=request.limit
        )

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error fetching data from Reddit : {e}")
    

    if reddit_df is None or reddit_df.empty:
        return _make_empty_response()
    

    if "text" not in reddit_df.columns:
        reddit_df["text"] = ""  

    reddit_df.loc[reddit_df["text"].fillna("").str.len() == 0, "text"] = reddit_df["title"].fillna("")

    rows = reddit_df.to_dict(orient="records")

    results = []
    for r in reddit_df.itertuples():
        text = r.text
        aspect = r.term
        title = r.title
        sentiment_result = analyze_sentiment(text, title, aspect)
        results.append(sentiment_result)


    for i, row in enumerate(rows):
        r = results[i]
        row["sentiment"] = r.get("label", "neutral")
        row["sentiment_score"] = float(r.get("score", 0.0))
        row["sarcasm_flag"] = bool(r.get("sarcasm_flag", False))


    analyzed_df = pd.DataFrame(rows)

    try:
        summary = summarize_df(analyzed_df)
    except Exception as e:
        summary = {
            "average_sentiment": float(analyzed_df["sentiment_score"].mean() if not analyzed_df.empty else 0.0),
            "trend" : {},
            "top_post": {},
            "most_positive_post": {},
            "most_negative_post": {}
        }

    safe_records = [_safe_record_from_row(row) for row in analyzed_df.to_dict(orient="records")]

    try:
        posts = [Post(**rec) for rec in safe_records]
    except Exception as e:
        import pdb
        pdb.set_trace()
        return JSONResponse(
            status_code=500, 
            content={
                "detail": f"Error processing data: {str(e)}",
                "posts": safe_records, 
                "summary": summary
            }
        )
    print(posts)
    return AnalyzeResponse(posts=posts, summary=summary)




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