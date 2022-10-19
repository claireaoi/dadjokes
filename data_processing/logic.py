from functools import cache
from typing import Iterable
import pandas as pd
import tweepy
import praw
from tweepy.models import Status as Tweet
from praw.models.reddit.submission import Submission as RedditPost
import yaml
import re

config = yaml.safe_load(open("secrets.yaml"))


@cache
def get_twitter_client():
    twitter_auth = tweepy.OAuth1UserHandler(
        config["twitter"]["API_key"],
        config["twitter"]["API_secret"],
        config["twitter"]["access_token"],
        config["twitter"]["access_secret"],
    )
    cli = tweepy.API(twitter_auth)
    return cli

def clean_joke(df):
    
    df1 = pd.DataFrame(columns=['joke','label','no_of_words'])
    df = df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    for i in range (len(df)):
        myString= str(df.iloc[i]['joke'])
        
        # remove white spaces retaining tab spaces.
        re.sub('\s+',' ',myString)
        re.sub('\n','',myString)
        #remove emoji,symbols
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
        emoji_pattern.sub(r'', myString)
        myString = ''.join(myString.splitlines())
        df1.at[i,'joke']= myString
        df1.at[i,'label']= str(df.iloc[i]['label'])
        df1.at[i,'no_of_words'] = len(myString)
        
    return df1

def get_tweets(user, limit):
    cli = get_twitter_client()
    tweets = tweepy.Cursor(
        cli.user_timeline, screen_name=user, tweet_mode="extended"
    ).items(limit)
    return list(tweets)


def tweets_to_df(tweets: Iterable[Tweet]):
    return pd.DataFrame(
        {
            # "obj": tweet,
            "id": tweet.id_str,
            "created_at": tweet.created_at,
            "source": f"tweeter/{tweet.author.screen_name}",
            "text": tweet.full_text,
            "url": (tweet.entities["urls"] or [{}])[0].get("display_url"),
        }
        for tweet in tweets
    )


@cache
def get_reddit_client():
    return praw.Reddit(
        client_id=config["reddit"]["client_id"],
        client_secret=config["reddit"]["client_secret"],
        user_agent="random snouglou",
    )


def get_reddits(subreddit_name, feed, limit):
    assert feed in ["hot", "new", "top"]
    cli = get_reddit_client()
    subreddit = cli.subreddit(subreddit_name)
    return list(getattr(subreddit, feed)(limit=limit))  # subreddit.feed(limit)


def reddits_to_df(submissions: Iterable[RedditPost]):
    return pd.DataFrame(
        {
            # "obj": subm,
            "id": subm.id,
            "created_at": subm.created_utc,
            "source": f"reddit/{subm.subreddit.display_name}",
            "text": subm.title,
            "content": subm.selftext,
            "url": subm.url,
        }
        for subm in submissions
    )


def reddits_to_jokes_df(
    submissions: Iterable[RedditPost],
    class_label: str = None,
    max_num_of_words: int = 70,
) -> pd.DataFrame:
    df = reddits_to_df(submissions)

    # Creating the new table with the jokes themselves
    new_df = pd.DataFrame(columns=["joke", "label", "joke_length_in_words"])

    # Building the joke column, being aware of the last character in the title
    new_df["joke"] = [None] * len(df)
    mask_for_last_character = df["text"].str[-1].isin([".", ",", "!", "?"])

    # If the last character is a period, comma, exclamation mark or question mark,
    # we concatenate with an empty space.
    new_df["joke"][mask_for_last_character] = df["text"] + " " + df["content"]

    # else, we add a period.
    new_df["joke"][~mask_for_last_character] = df["text"] + ". " + df["content"]

    new_df["label"] = class_label
    new_df["joke_length_in_words"] = new_df["joke"].apply(
        lambda joke: len(joke.split(" "))
    )
    new_df = new_df[new_df["joke_length_in_words"] <= max_num_of_words]

    return new_df
