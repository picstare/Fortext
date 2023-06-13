import os
import streamlit as st
import tweepy
from streamlit_tags import st_tags
import json
import re
from datetime import datetime

from json import JSONEncoder



# # Define the User class
# class User:
#     def __init__(self, id, name, username):
#         self.id = id
#         self.name = name
#         self.username = username

class DateTimeEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

# Set up Tweepy client
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAN2QoAEAAAAAVJDVMnIlp5SOzQ7sDF9whqRk5zk%3DYJR7rwXSh1utXkl6thLwejvqlJh1kiyNpP3izvP5l3F93iJVfo'
client = tweepy.Client(bearer_token=bearer_token)
retweeted_tweet_id = None  # Declare the variable outside the if block

# Create "twitkey" folder if it doesn't exist
if not os.path.exists("twitkey"):
    os.makedirs("twitkey")

# Streamlit app title
st.title("Twitter Search")

# Create a form using st.form
with st.form(key="tkeysform"):
    # Add tag input for keywords
    keywords = st_tags(
        label='# Enter Keywords:',
        text='Press enter to add more',
        value=[],
        suggestions=[],
        maxtags=4,
        key='2'
    )

    # Add search button within the form
    search_button = st.form_submit_button(label="Search")

if search_button and keywords:
    for keyword in keywords:
        results = []

        # Perform search for each keyword
        tweets = client.search_recent_tweets(
            query=keyword,
            tweet_fields=[
                'context_annotations', 'text','created_at', 'entities', 'source', 'geo', 'public_metrics', 'referenced_tweets'
            ],
            user_fields=['name', 'username', 'profile_image_url', 'description', 'location'],
            expansions=[
                'author_id', 'referenced_tweets.id', 'referenced_tweets.id.author_id',
                'in_reply_to_user_id', 'entities.mentions.username', 'geo.place_id'
            ],
            max_results=5
        )

        # Get users list from the includes object
        users = {u["id"]: u for u in tweets.includes['users']}

        for tweet in tweets.data:
            # Initialize 'place_name' and 'in_reply_to_name' variables
            place_name = ''
            in_reply_to_name = ''

            user_id = tweet.author_id
            if user_id in users:
                user = users[user_id]
                user_name = user['name']
                user_screen_name = user['username']
                profile_image_url = user['profile_image_url']
                user_description = user['description']
                user_location = user.get('location', None)

                # Extract retweet, mention, and reply information
                retweet_count = tweet.public_metrics['retweet_count']
                reply_count = tweet.public_metrics['reply_count']
                mention_count = 0

                mentioned_users = []
                if 'entities' in tweet and 'in_reply_to_user_id' in tweet.entities:
                    in_reply_to_user_id = tweet.entities['in_reply_to_user_id']
                    if in_reply_to_user_id in users:
                        in_reply_to_user = users[in_reply_to_user_id]
                        in_reply_to_name = in_reply_to_user['name']

                # Get like count and quote count
                like_count = tweet.public_metrics['like_count']
                quote_count = tweet.public_metrics['quote_count']

                referenced_tweet = None
                if 'referenced_tweets' in tweet:
                    referenced_tweet_id = tweet.referenced_tweets[0]['id']
                    referenced_tweet = next(
                        (t for t in tweets.includes['tweets'] if t['id'] == referenced_tweet_id), None
                    )

                referenced_tweet_text = referenced_tweet['text'] if referenced_tweet else None

                if 'entities' in tweet and 'hashtags' in tweet.entities:
                    hashtags = [tag['tag'] for tag in tweet.entities['hashtags']]
                else:
                    hashtags = []

                full_text=tweet.text

                # Extract relevant fields from tweet and user
                tweet_data = {
                    'User Name': user_name,
                    'User Screen Name': user_screen_name,
                    'Profile Image URL': profile_image_url,
                    'User Description': user_description,
                    'User Location': user_location if user_location else None,
                    'Created At': tweet.created_at,
                    'Tweet ID': tweet.id,
                    'Text': full_text,
                    'Hashtags': hashtags,
                    'Tweet URL': f"https://twitter.com/{user_screen_name}/status/{tweet.id}",
                    'Source': tweet.source or '',
                    'Retweet Count': retweet_count,
                    'Reply Count': reply_count,
                    'Mention Count': mention_count,
                    'in_reply_to_name': in_reply_to_name if in_reply_to_name else None,
                    'mentioned_users': mentioned_users if mentioned_users else [],
                }

                

                if 'referenced_tweets' in tweet:
                    referenced_tweet = next(
                        (t for t in tweets.includes['tweets'] if t['id'] == referenced_tweet_id), None
                    )

                    if referenced_tweet and referenced_tweet.get('type') == 'retweeted':
                        retweeted_tweet_id = referenced_tweet['id']
                        retweets = client.get_tweet_retweets(id=retweeted_tweet_id, max_results=10)

                        # Extract retweets' information
                        retweet_data = []
                        for retweet in retweets.data:
                            retweet_id = retweet['id']
                            retweeter_id = retweet['author_id']

                            if retweeter_id in users:
                                retweeter_user = users[retweeter_id]
                                retweeter_name = retweeter_user['name']
                                retweeter_screen_name = retweeter_user['username']
                                retweeter_profile_image_url = retweeter_user['profile_image_url']

                                retweet_data.append({
                                    'Retweet ID': retweet_id,
                                    'Retweeter Name': retweeter_name,
                                    'Retweeter Screen Name': retweeter_screen_name,
                                    'Retweeter Profile Image URL': retweeter_profile_image_url,
                                })

                        tweet_data['Retweets'] = retweet_data

                # Move the following code outside the if block
                if retweeted_tweet_id:
                    retweeters = client.get_retweeters(id=retweeted_tweet_id, max_results=10)

                    # Extract retweeters' information
                    retweeter_data = []
                    for retweeter in retweeters.data:
                        retweeter_id = retweeter['id']
                        if retweeter_id in users:
                            retweeter_user = users[retweeter_id]
                            retweeter_name = retweeter_user['name']
                            retweeter_screen_name = retweeter_user['username']
                            retweeter_profile_image_url = retweeter_user['profile_image_url']

                            retweeter_data.append({
                                'Retweeter Name': retweeter_name,
                                'Retweeter Screen Name': retweeter_screen_name,
                                'Retweeter Profile Image URL': retweeter_profile_image_url,
                            })

                    tweet_data['Retweeters'] = retweeter_data

                results.append(tweet_data)

        # Create a directory if it doesn't exist
        os.makedirs("twitkeys", exist_ok=True)

        # Save the results in a JSON file named with the keyword
        output = {"data": results}
        file_path = os.path.join("twitkeys", f"{keyword}.json")
        with open(file_path, 'w') as json_file:
            json.dump(output, json_file, cls=DateTimeEncoder)