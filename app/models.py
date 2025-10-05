from typing import Optional, Dict, List, Literal
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):

    text :  str = Field(..., description="Keyword/Aspect to analyze"  )
    subreddit : str = Field ('all', description = "Name of Subreddit or 'all'")
    limit : int = Field(100, description = "Max no of posts to fetch, defualts to 100")



