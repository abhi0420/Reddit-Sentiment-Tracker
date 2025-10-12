from typing import Optional, Dict, List, Literal
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):

    text :  str = Field(..., description="Keyword/Aspect to analyze"  )
    subreddit : str = Field ('all', description = "Name of Subreddit or 'all'")
    limit : int = Field(100, description = "Max no of posts to fetch, defualts to 100")


class Sentiment(BaseModel):
        label : Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
        score : float = Field(..., ge = -1.0, le = 1.0, description="Sentiment score")

class Sarcasm(BaseModel):
    flag : bool
    confidence : float = Field(0.0, ge = 0.0, le = 1.0, description="Confidence score for sarcasm detection")


class Post(BaseModel):
    id : str
    created_on_utc : str
    term : str
    subreddit : str
    author : Optional[str]
    title : str
    text : str
    score : int
    url : str
    is_self : bool
    permalink : str

    sentiment : Optional[float]
    sentiment_score : Optional[float]
    sarcasm_flag : Optional[bool]
        

class Summary(BaseModel):
    average_sentiment : float
    trend : Dict[str, float]
    top_post : Dict[str, object]
    most_positive_post : Dict[str, object]
    most_negative_post : Dict[str, object]

class AnalyzeResponse(BaseModel):
    posts : List[Post]
    summary : Summary