from logic import (
    get_reddits,
    reddits_to_df,
    get_tweets,
    reddits_to_jokes_df,
    tweets_to_df,
)
import click

import pandas as pd


@click.group()
def cli():
    pass


@cli.command()
@click.argument("output", type=click.File("wb"))
@click.argument("subreddits", nargs=-1)
@click.option("--feed", default="hot")
@click.option("--limit", type=int, default=100)
@click.option("--max-num-of-words", type=int, default=70)
def reddit(output, subreddits, feed, limit, max_num_of_words):
    reddits = []
    for subr in subreddits:
        reddits += get_reddits(subr, feed, limit)
    df = reddits_to_df(reddits)

    # Creating the new table with the jokes themselves
    new_df = pd.DataFrame(columns=["joke", "label", "joke_length_in_words"])
    new_df["joke"] = df["text"] + " " + df["content"]
    new_df["label"] = subreddits[0]
    new_df["joke_length_in_words"] = new_df["joke"].apply(
        lambda joke: len(joke.split(" "))
    )
    new_df = new_df[new_df["joke_length_in_words"] <= max_num_of_words]
    print(new_df)
    new_df.to_csv(output)


@cli.command()
@click.argument("output", type=click.File("wb"))
@click.option("--limit", type=int, default=100)
@click.option("--max-num-of-words", type=int, default=70)
def jokes(output, limit, max_num_of_words):
    reddits = []

    # Requesting req.limit of dad jokes and jokes
    dfs = []
    for subreddit in ["dadjokes", "jokes"]:
        for feed in ["hot", "new", "top"]:
            posts = get_reddits(subreddit, feed=feed, limit=limit)
            dfs.append(
                reddits_to_jokes_df(posts, subreddit, max_num_of_words=max_num_of_words)
            )

    # Concatenating both dfs
    df = pd.concat(dfs, ignore_index=True)

    df.to_csv(output)


@cli.command()
@click.argument("output", type=click.File("wb"))
@click.argument("accounts", nargs=-1)
@click.option("--limit", type=int, default=100)
def twitter(output, accounts, limit):
    tweets = []
    for acc in accounts:
        tweets += get_tweets(acc, limit)
    df = tweets_to_df(tweets)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    cli()
