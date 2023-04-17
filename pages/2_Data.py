import time
import streamlit as st
import pandas as pd
import numpy as np
import notebook
from datetime import datetime, date, timedelta
import snscrape.modules.twitter as sntwit
# from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import re
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px
from  PIL import Image
from pyvis.network import Network
import matplotlib.pyplot as plt
import scipy as sp
import streamlit.components.v1 as components
# from streamlit_tags import st_tags
import altair as alt
import seaborn as sns
from tokenisasi import Tokenizer
import functools
from genderpred import GndrPrdct
from math import sin
import sys
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import WordNetLemmatizer
from googletrans import Translator, constants
# import streamlit_nested_layout
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import agepred
# from agepred import get_age
from collections import OrderedDict
from networkx.algorithms import bipartite
from networkx import NetworkXError
import sys
import pickle
import gensim
from gensim.models.ldamodel import LdaModel
# import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from pprint import pprint
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models
import pyLDAvis.gensim
from streamlit_authenticator import Authenticate, authenticate
import yaml
import Home
from yaml.loader import SafeLoader
from Home import authenticator
from streamlit_extras.switch_page_button import switch_page



sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'         # define a string of punctuation symbols

# model_path = "sklearn/lda_sklearn.pkl"
# vectorizer_path = "sklearn/vectorizer.pkl"

st.set_page_config(
    page_title="Data.Picanalytics",
    layout="wide"
      
)

if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'sidebar')
    st.write(f'Welcome *{st.session_state["name"]}*')
# elif st.session_state["authentication_status"] is False:
#     st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    # st.warning('Please enter your username and password')
    switch_page('Home')

# with open('config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)


# authenticator = Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )

st.write('<base target="_blank">', unsafe_allow_html=True)
a, b = st.columns([1, 10])



with a:
    st.text("")
    st.image("img/twitterlkogo.png", width=50)
with b:
    st.title("Twitter Analyzer")

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div.vis-configuration-wrapper{display: block; width: 400px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)



with st.container():
    attributes_container = []
    outer_cols_a, outer_cols_b = st.columns([4, 2])
    sentimentweet = SentimentIntensityAnalyzer()

    with outer_cols_b:
        with st.form(key="tkeywordform"):
            keywords=st.text_input("Keywords","")
            # keywords = st_tags(
            #     label='# Enter Keywords:',
            #     text='Press enter to add more',
            #     value=['Zero', 'One', 'Two'],suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'], 
            #     maxtags = 4, key='tweetkeywords')
            
            username=st.text_input("User Name","")
            # keywordstr=' OR '.join(map(str,keywords))
            # keywordfind=' | '.join(map(str,keywords))
            defaultenddate = date.today()
            defaultstartdate = defaultenddate - timedelta(days=7)

            inner_cols_a, inner_cols_b = st.columns([1, 1])
            since=inner_cols_a.date_input("Start date", defaultstartdate)
            until=inner_cols_b.date_input("End date", defaultenddate)
            # sdatestr = startdate.strftime("YYYY-MM-DD")
            # edatestr = enddate.strftime("YYYY-MM-DD")
            
            inner_cols_c,inner_cols_d, inner_cols_e=st.columns(3)
            maxTweets = inner_cols_c.number_input('Insert a number', 0)
            retweets= inner_cols_d.checkbox("Exclude retweets", False)
            replies=inner_cols_e.checkbox("Exclude replies", False)

            submit= st.form_submit_button(label="Submit")


            def search(keywords,username,since,until,retweets,replies):
                global filename
                q = keywords
                if username!='':
                    q += f" from:{username}"   
                if since!='':
                    q += f" since:{since}"
                if until!='':
                    q += f" until:{until}"
                if retweets == True:
                    q += f" exclude:retweet"
                if replies == True:
                    q += f" exclude:replies"
                if username!='' and keywords!='':
                    filename = f"{since}_{until}_{username}_{keywords}.pkl"
                elif username!="":
                    filename = f"{since}_{until}_{username}.pkl"
                else:
                    filename = f"{since}_{until}_{keywords}.pkl"
                # print(filename)
                return q
            q = search(keywords,username,since,until,retweets,replies)

            
            if submit:
                for i,tweet in enumerate(sntwit.TwitterSearchScraper(q, top = True).get_items()):
                    if i>=maxTweets: #number of tweets you want to scrape
                        break
                    attributes_container.append([tweet.date, tweet.id, tweet.rawContent, tweet.user.username, tweet.user.profileImageUrl, tweet.user.followersCount, tweet.user.listedCount, 
                                                    tweet.user.location, tweet.sourceLabel, tweet.lang, tweet.replyCount,tweet.retweetCount,tweet.likeCount,tweet.quoteCount, tweet.media ])

        tweets_df = pd.DataFrame(attributes_container, columns=['DateTime', 'TweetId', 'Text', 'Username','ProfileImageUrl', 'Followers','Listed', 'Location', 'Device', 'Language',
                            'ReplyCount','RetweetCount','LikeCount','QuoteCount', 'Media'])
        
        # st.write(q)
        

        tweets_df.sort_values(by='DateTime',ascending=False)


        def find_ret(text):
            rtweet=re.findall(r'^(?=.*\bRT\b\s@\b).*$', text)
            return ",".join(rtweet)
        tweets_df['Retweets']=tweets_df['Text'].apply(lambda x: find_ret(x))
        tweets_df['Splitretweet'] =tweets_df['Retweets'].apply(lambda x: x.split(','))
    
        def find_mention(text):
            mentname=re.findall(r'(?<![@\w])@(\w{1,25})',text)
            return ",".join(mentname)
        
        def find_hashtag(text):
            mentname=re.findall(r'(?<=#)\w+',text)
            return ",".join(mentname)
        tweets_df['Hashtags'] = tweets_df['Text'].apply(lambda x: find_hashtag(x))
        tweets_df['Splithast'] = tweets_df['Hashtags'].apply(lambda x: x.split(','))

        
        tweets_df['Mentions'] = tweets_df['Text'].apply(lambda x: find_mention(x))
        tweets_df['Splitmentions'] = tweets_df['Mentions'].apply(lambda x: x.split(','))

        ##########################
        # tweets_df['Splithashtags'] = tweets_df['Hashtags'].apply(lambda x: x.split(','))

        ##########################

        def remove_links(tweet):
            """Takes a string and removes web links from it"""
            tweet = re.sub(r'http\S+', '', tweet)   # remove http links
            tweet = re.sub(r'bit.ly/\S+', '', tweet)  # remove bitly links
            tweet = tweet.strip('[link]')   # remove [links]
            tweet = re.sub(r'pic.twitter\S+','', tweet)
            return tweet


        tweets_df['Nourl'] = tweets_df['Text'].apply(lambda x: remove_links(x))

    

        #########TRANSLATION############

        translator = Translator()
        
        tweets_df['Tweeten'] = tweets_df['Nourl'].apply(lambda x: translator.translate(x, dest='en').text)

        

        #############SENTIMENT#########################

        def vader_sentiment(text):
            return sentimentweet.polarity_scores(text)['compound']

        # create new column for vadar compound sentiment score
        tweets_df['Compoundscore'] = tweets_df['Tweeten'].apply(lambda x: vader_sentiment(x))
        

        def catsentiment(sentiment, neg_threshold=-0.05, pos_threshold=0.05):
            """ categorise the sentiment value as positive (1), negative (-1) 
            or neutral (0) based on given thresholds """
            if sentiment < neg_threshold:
                label = 'negative'
            elif sentiment > pos_threshold:
                label = 'positive'
            else:
                label = 'neutral'
            return label

        # new col with vadar sentiment label based on vadar compound score
        tweets_df['Sentiment'] = tweets_df['Compoundscore'].apply(lambda x: catsentiment(x))

        ################GENDERPREDICTION#######

        Classifier  = GndrPrdct()

        tweets_df['Gender'] = tweets_df['Tweeten'].apply(lambda x: Classifier.predict_gender(x))

        def map_gender(sex):
            if sex == "female":
                return "#A459D1"
            elif sex == "male":
                return "#FFB84C"

        tweets_df['Colorgend']=tweets_df['Gender'].apply(lambda sex: map_gender(sex)) 

        ################AGEPREDICTION####################

        # tweets_df['Ages'] = tweets_df['Text'].apply(lambda x: agepred.get_age(x))


        #################################################
    
        def remove_users(tweet):
            """Takes a string and removes retweet and @user information"""
            tweet = re.sub('(RT\\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove re-tweet
            tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove tweeted at
            return tweet
        
        
        def remove_hashtags(tweet):
            """Takes a string and removes any hash tags"""
            tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove hash tags
            return tweet
        

        def remove_av(tweet):
            """Takes a string and removes AUDIO/VIDEO tags or labels"""
            tweet = re.sub('VIDEO:', '', tweet)  # remove 'VIDEO:' from start of tweet
            tweet = re.sub('AUDIO:', '', tweet)  # remove 'AUDIO:' from start of tweet
            return tweet
        
    
        def tokenize(tweet):
            """Returns tokenized representation of words in lemma form excluding stopwords"""
            result = []
            for token in gensim.utils.simple_preprocess(tweet):
                if token not in gensim.parsing.preprocessing.STOPWORDS \
                        and len(token) > 2:  # drops words with less than 3 characters
                    result.append(lemmatize(token))
            return result

        def lemmatize(token):
            """Returns lemmatization of a token"""
            return WordNetLemmatizer().lemmatize(token, pos='v')
        
    
        def preprocess_tweet(tweet):
            """Main master function to clean tweets, stripping noisy characters, and tokenizing use lemmatization"""
            tweet = remove_users(tweet)
            tweet = remove_hashtags(tweet)
            tweet = remove_av(tweet)
            tweet = tweet.lower()  # lower case
            tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
            tweet = re.sub("\\s\\s+", ' ', tweet)  # remove double spacing
            tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
            tweet_token_list = tokenize(tweet)  # apply lemmatization and tokenization
            tweet = ' '.join(tweet_token_list)
            return tweet
        
        
    #     def basic_clean(tweet):
    #         """Main master function to clean tweets only without tokenization or removal of stopwords"""
    #         tweet = remove_users(tweet)
    #         tweet = remove_links(tweet)
    #         tweet = remove_hashtags(tweet)
    #         tweet = remove_av(tweet)
    #         tweet = tweet.lower()  # lower case
    #         tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    #         tweet = re.sub('\\s+', ' ', tweet)  # remove double spacing
    #         tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    #         tweet = re.sub('📝 …', '', tweet)
    #         return tweet
        
        
        
        def tokenize_tweets(df):
            """Main function to read in and return cleaned and preprocessed dataframe.
            This can be used in Jupyter notebooks by importing this module and calling the tokenize_tweets() function
            Args:
                df = data frame object to apply cleaning to
            Returns:
                pandas data frame with cleaned tokens
            """

            df['tokens'] = tweets_df.Tweeten.apply(preprocess_tweet)
            num_tweets = len(df)
            print('Complete. Number of Tweets that have been cleaned and tokenized : {}'.format(num_tweets))
            return df
        
        tweets_df = tokenize_tweets(tweets_df)

        

        tweets_df['DateTime'] = pd.to_datetime(tweets_df['DateTime'], unit='ms')


        tweets_df['Hour'] = tweets_df['DateTime'].dt.hour
        tweets_df['Year'] = tweets_df['DateTime'].dt.year  
        tweets_df['Month'] = tweets_df['DateTime'].dt.month
        tweets_df['MonthName'] = tweets_df['DateTime'].dt.month_name()
        tweets_df['MonthDay'] = tweets_df['DateTime'].dt.day
        tweets_df['DayName'] = tweets_df['DateTime'].dt.day_name()
        tweets_df['Week'] = tweets_df['DateTime'].dt.isocalendar().week

        tweets_df['Date'] = [d.date() for d in tweets_df['DateTime']]
        tweets_df['Time'] = [d.time() for d in tweets_df['DateTime']]

        tweets_df.drop('DateTime',axis=1,inplace=True)
        tweets_df.drop('Retweets',axis=1,inplace=True)
        tweets_df.drop('Mentions',axis=1,inplace=True)
        # tweets_df.drop('Mentions',axis=1,inplace=True)
        tweets_df.drop('Hashtags',axis=1,inplace=True)
    


        # if tweets_df.empty:
        #     st.write('')
        # else:
        #     tweets_df.info()

            # st.dataframe(tweets_df)
            
            # tweets_df.to_csv(f"{filename}",index=True)
            # tweets_df.to_pickle('data/dummy.pkl', compression='infer', protocol=4)

datapath = "data"   
            
with outer_cols_a:
    if tweets_df.empty:
        st.write('')
    else:
        tweets_df.info()
        st.dataframe(tweets_df)
        tweets_df.to_pickle(f'{datapath}/{filename}', compression='infer', protocol=4)

    
