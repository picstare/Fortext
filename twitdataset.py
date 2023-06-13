# import os
# import streamlit as st
# # import tweepy
# import pandas as pd
# import base64


from parsel import Selector
from playwright.sync_api import sync_playwright
from playwright.sync_api._generated import Page

def parse_tweets(selector: Selector):
    """
    parse tweets from pages containing tweets like:
    - tweet page
    - search page
    - reply page
    - homepage
    returns list of tweets on the page where 1st tweet is the 
    main tweet and the rest are replies
    """
    results = []
    # select all tweets on the page as individual boxes
    # each tweet is stored under <article data-testid="tweet"> box:
    tweets = selector.xpath("//article[@data-testid='tweet']")
    for i, tweet in enumerate(tweets):
        # using data-testid attribute we can get tweet details:
        found = {
            "text": "".join(tweet.xpath(".//*[@data-testid='tweetText']//text()").getall()),
            "username": tweet.xpath(".//*[@data-testid='User-Names']/div[1]//text()").get(),
            "handle": tweet.xpath(".//*[@data-testid='User-Names']/div[2]//text()").get(),
            "datetime": tweet.xpath(".//time/@datetime").get(),
            "verified": bool(tweet.xpath(".//svg[@data-testid='icon-verified']")),
            "url": tweet.xpath(".//time/../@href").get(),
            "image": tweet.xpath(".//*[@data-testid='tweetPhoto']/img/@src").get(),
            "video": tweet.xpath(".//video/@src").get(),
            "video_thumb": tweet.xpath(".//video/@poster").get(),
            "likes": tweet.xpath(".//*[@data-testid='like']//text()").get(),
            "retweets": tweet.xpath(".//*[@data-testid='retweet']//text()").get(),
            "replies": tweet.xpath(".//*[@data-testid='reply']//text()").get(),
            "views": (tweet.xpath(".//*[contains(@aria-label,'Views')]").re("(\d+) Views") or [None])[0],
        }
        # main tweet (not a reply):
        if i == 0:
            found["views"] = tweet.xpath('.//span[contains(text(),"Views")]/../preceding-sibling::div//text()').get()
            found["retweets"] = tweet.xpath('.//a[contains(@href,"retweets")]//text()').get()
            found["quote_tweets"] = tweet.xpath('.//a[contains(@href,"retweets/with_comments")]//text()').get()
            found["likes"] = tweet.xpath('.//a[contains(@href,"likes")]//text()').get()
        results.append({k: v for k, v in found.items() if v is not None})
    return results

def parse_profiles(sel: Selector):
    """parse profile preview data from Twitter profile search"""
    profiles = []
    for profile in sel.xpath("//div[@data-testid='UserCell']"):
        profiles.append(
            {
                "name": profile.xpath(".//a[not(@tabindex=-1)]//text()").get().strip(),
                "handle": profile.xpath(".//a[@tabindex=-1]//text()").get().strip(),
                "bio": ''.join(profile.xpath("(.//div[@dir='auto'])[last()]//text()").getall()),
                "url": profile.xpath(".//a/@href").get(),
                "image": profile.xpath(".//img/@src").get(),
            }
        )
    return profiles


def scrape_top_search(query: str, page: Page):
    """scrape top Twitter page for featured tweets"""
    page.goto(f"https://twitter.com/search?q={query}&src=typed_query")
    page.wait_for_selector("//article[@data-testid='tweet']")  # wait for content to load
    tweets = parse_tweets(Selector(page.content()))
    return tweets


def scrape_people_search(query: str, page: Page):
    """scrape people search Twitter page for related users"""
    page.goto(f"https://twitter.com/search?q={query}&src=typed_query&f=user")
    page.wait_for_selector("//div[@data-testid='UserCell']")  # wait for content to load
    profiles = parse_profiles(Selector(page.content()))
    return profiles


with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    
    top_tweet_search = scrape_top_search("google", page)
    people_tweet_search = scrape_people_search("google", page)

# # Authenticate with Twitter API using OAuth1UserHandler
# consumer_key = "wtph1D9eE27h2pTwAfUUZFJGh"
# consumer_secret = "ueIvSjFVV6MkH7yKtC67ybi6qkPiV4xJun4CsBv8w22lwY6eTF"
# access_token = "16645853-1WS14NgT2p9m7sMH3s7xU4G5QRN2YBRFXXEYjgEnd"
# access_token_secret = "Csj5OhyNTUAZOxkBsWi9d7GHnwbQHkLIFgowiBda6lM1o"

# auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
# api = tweepy.API(auth)

# # Function to get tweets based on query keyword
# def get_tweets(query, count):
#     tweets = []
#     try:
#         for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="id").items(count):
#             parsed_tweet = {}
#             parsed_tweet["text"] = tweet.text
#             parsed_tweet["sentiment"] = ""
#             tweets.append(parsed_tweet)
#         return tweets
#     except tweepy.TweepyException as e:
#         print("Error : " + str(e))

# # Main function to run Streamlit app
# def main():
#     st.title("Twitter Sentiment Analysis Dataset Creator")

#     # Get user input for query keyword and number of tweets
#     query = st.text_input("Enter keyword for query", "COVID-19")
#     count = st.number_input("Enter number of tweets to fetch", 100, 10000, 100)

#     # Fetch tweets and display them in a table
#     if st.button("Fetch Tweets"):
#         st.write("Fetching tweets...")
#         tweets = get_tweets(query, count)
#         df = pd.DataFrame(tweets)
        
#         # Check if directory exists and create it if not
#         if not os.path.exists("tdset"):
#             os.mkdir("tdset")
        
#         # Check if file exists and append or create new file
#         if os.path.exists("tdset/tsds.csv"):
#             existing_data = pd.read_csv("tdset/tsds.csv")
#             new_data = existing_data.append(df, ignore_index=True)#DEPRECATED
#             new_data.to_csv("tdset/tsds.csv", index=False)
#             st.write(f"Added {count} tweets to existing file: tdset/tsds.csv")
#         else:
#             df.to_csv("tdset/tsds.csv", index=False)
#             st.write(f"Created new file: tdset/tsds.csv")

#         st.write(df)

#         # Allow user to download the data as CSV file
#         csv = df.to_csv(index=False)
#         b64 = base64.b64encode(csv.encode()).decode()
#         href = f'<a href="data:file/csv;base64,{b64}" download="tweets.csv">Download CSV File</a>'
#         st.markdown(href, unsafe_allow_html=True)

# # Call the main function
# main()