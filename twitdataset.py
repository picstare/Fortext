import os
import streamlit as st
import tweepy
import pandas as pd
import base64

# Authenticate with Twitter API using OAuth1UserHandler
consumer_key = "wtph1D9eE27h2pTwAfUUZFJGh"
consumer_secret = "ueIvSjFVV6MkH7yKtC67ybi6qkPiV4xJun4CsBv8w22lwY6eTF"
access_token = "16645853-1WS14NgT2p9m7sMH3s7xU4G5QRN2YBRFXXEYjgEnd"
access_token_secret = "Csj5OhyNTUAZOxkBsWi9d7GHnwbQHkLIFgowiBda6lM1o"

auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Function to get tweets based on query keyword
def get_tweets(query, count):
    tweets = []
    try:
        for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="id").items(count):
            parsed_tweet = {}
            parsed_tweet["text"] = tweet.text
            parsed_tweet["sentiment"] = ""
            tweets.append(parsed_tweet)
        return tweets
    except tweepy.TweepyException as e:
        print("Error : " + str(e))

# Main function to run Streamlit app
def main():
    st.title("Twitter Sentiment Analysis Dataset Creator")

    # Get user input for query keyword and number of tweets
    query = st.text_input("Enter keyword for query", "COVID-19")
    count = st.number_input("Enter number of tweets to fetch", 100, 10000, 100)

    # Fetch tweets and display them in a table
    if st.button("Fetch Tweets"):
        st.write("Fetching tweets...")
        tweets = get_tweets(query, count)
        df = pd.DataFrame(tweets)
        
        # Check if directory exists and create it if not
        if not os.path.exists("tdset"):
            os.mkdir("tdset")
        
        # Check if file exists and append or create new file
        if os.path.exists("tdset/tsds.csv"):
            existing_data = pd.read_csv("tdset/tsds.csv")
            new_data = existing_data.append(df, ignore_index=True)#DEPRECATED
            new_data.to_csv("tdset/tsds.csv", index=False)
            st.write(f"Added {count} tweets to existing file: tdset/tsds.csv")
        else:
            df.to_csv("tdset/tsds.csv", index=False)
            st.write(f"Created new file: tdset/tsds.csv")

        st.write(df)

        # Allow user to download the data as CSV file
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="tweets.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

# Call the main function
main()