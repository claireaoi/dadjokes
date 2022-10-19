from typing import List
from pydantic import BaseModel
from fastapi import FastAPI

from logic import get_reddits, reddits_to_jokes_df, get_tweets, tweets_to_df

import pandas as pd

app = FastAPI()


class RedditRequest(BaseModel):
    subreddits: List[str]
    feed: str = "hot"
    limit: int = 100
    dest: str


class JokesRequest(BaseModel):
    limit: int = 100
    max_num_of_words: int = 70
    dest: str


@app.post("/reddit")
def api_reddit(req: RedditRequest):
    try:
        reddits = []
        for subr in req.subreddits:
            reddits += get_reddits(subr, req.feed, req.limit)
        df = reddits_to_jokes_df(reddits, req.subreddits[0])

        df.to_gbq(
            req.dest
        )  # write to the Big Query bucket specified by dest. change if you want some other behavior.
    except Exception as e:
        return repr(e)
    return "OK"


@app.post("/jokes")
def api_jokes(req: JokesRequest):
    """
    Requests req.limit jokes for each subreddit r/Jokes and r/DadJokes,
    since we're filtering by maximum number of words, we might not get
    exactly req.limit
    """
    try:
        # Requesting req.limit of dad jokes and jokes
        dfs = []
        for subreddit in ["dadjokes", "jokes"]:
            posts = get_reddits(subreddit, feed="hot", limit=req.limit)
            dfs.append(
                reddits_to_jokes_df(
                    posts, subreddit, max_num_of_words=req.max_num_of_words
                )
            )

        # Concatenating both dfs
        df = pd.concat(dfs)

        # Sending to BigQuery.
        df.to_gbq(req.dest)
    except Exception as e:
        return repr(e)
    return "OK"


class TwitterRequest(BaseModel):
    accounts: List[str]
    limit: int = 100
    dest: str


@app.post("/twitter")
def api_twitter(req: TwitterRequest):
    try:
        tweets = []
        for acc in req.accounts:
            tweets += get_tweets(acc, req.limit)
        df = tweets_to_df(tweets)
        df.to_gbq(req.dest)
    except Exception as e:
        return repr(e)
    return "OK"
