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

choose=st.radio(
    "Analyze by",
    ('Keywords', 'Hastags', 'Users', 'File'))

if choose == 'Keywords':

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
                    for i,tweet in enumerate(sntwit.TwitterSearchScraper(q).get_items()):
                        if i>=maxTweets: #number of tweets you want to scrape
                            break
                        attributes_container.append([tweet.date, tweet.id, tweet.rawContent, tweet.user.username, tweet.user.profileImageUrl, tweet.user.followersCount, tweet.user.listedCount, 
                                                     tweet.user.location, tweet.sourceLabel, tweet.lang, tweet.replyCount,tweet.retweetCount,tweet.likeCount,tweet.quoteCount, tweet.media ])

            tweets_df = pd.DataFrame(attributes_container, columns=['DateTime', 'TweetId', 'Text', 'Username','ProfileImageUrl', 'Followers','Listed', 'Location', 'Device', 'Language',
                                'ReplyCount','RetweetCount','LikeCount','QuoteCount', 'Media'])
            
            st.write(q)
            

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

            # tweets_df.drop('DateTime',axis=1,inplace=True)
            tweets_df.drop('Retweets',axis=1,inplace=True)
            tweets_df.drop('Mentions',axis=1,inplace=True)
            # tweets_df.drop('Mentions',axis=1,inplace=True)
            tweets_df.drop('Hashtags',axis=1,inplace=True)
        


            if tweets_df.empty:
                st.write('')
            else:
                tweets_df.info()

                st.dataframe(tweets_df)
                tweets_df.to_pickle(f'{filename}', compression='infer', protocol=4)
                # tweets_df.to_csv(f"{filename}",index=True)
                # tweets_df.to_pickle('data/dummy.pkl', compression='infer', protocol=4)

        
                
    with outer_cols_a:
        st.write('')
        
        # tweets_df1 = pd.read_csv("2023-02-01_2023-02-28_psi_id_kopdarnas2.csv")
        

        # tweets_df1.sort_values(by='DateTime',ascending=False)
        
        # def find_mention(text):
        #     mentname=re.findall(r'(?<![@\w])@(\w{1,25})',text)
        #     return ",".join(mentname)
        
        # tweets_df1['Mentions'] = tweets_df1['Text'].apply(lambda x: find_mention(x).split(','))

        # ##########################

        # def remove_links(tweet):
        #     """Takes a string and removes web links from it"""
        #     tweet = re.sub(r'http\S+', '', tweet)   # remove http links
        #     tweet = re.sub(r'bit.ly/\S+', '', tweet)  # remove bitly links
        #     tweet = tweet.strip('[link]')   # remove [links]
        #     tweet = re.sub(r'pic.twitter\S+','', tweet)
        #     # tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
        #     # tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove hash tags
        #     # tweet = re.sub("\\s\\s+", ' ', tweet)
        #     return tweet


        # tweets_df1['Nourl'] = tweets_df1['Text'].apply(lambda x: remove_links(x))

       

        # #########TRANSLATION############

        # translator = Translator()
        
        # tweets_df1['Tweeten'] = tweets_df1['Nourl'].apply(lambda x: translator.translate(x, dest='en', src='id').text)

        

        # #############SENTIMENT#########################

        # def vader_sentiment(text):
        #     return sentimentweet.polarity_scores(text)['compound']

        # # create new column for vadar compound sentiment score
        # tweets_df1['Compoundscore'] = tweets_df1['Tweeten'].apply(lambda x: vader_sentiment(x))
        

        # def catsentiment(sentiment, neg_threshold=-0.05, pos_threshold=0.05):
        #     """ categorise the sentiment value as positive (1), negative (-1) 
        #     or neutral (0) based on given thresholds """
        #     if sentiment < neg_threshold:
        #         label = 'negative'
        #     elif sentiment > pos_threshold:
        #         label = 'positive'
        #     else:
        #         label = 'neutral'
        #     return label

        # # new col with vadar sentiment label based on vadar compound score
        # tweets_df1['Sentiment'] = tweets_df1['Compoundscore'].apply(lambda x: catsentiment(x))

        # tweets_df.info()

        # st.dataframe(tweets_df1)

        


        


       

        
            
            
            





#                     for i, tweet in enumerate(sntwit.TwitterSearchScraper( keywordstr + ' since:%s until:%s ' %(startdate, enddate), top = True ).get_items()):
#                         if i>maxTweets:
#                             break
#                         attributes_container.append([tweet.id, tweet.conversationId, tweet.user.username, tweet.user.displayname, tweet.user.description, tweet.date, tweet.user.profileImageUrl, tweet.user.followersCount, tweet.likeCount, tweet.retweetCount, tweet.user.listedCount, tweet.sourceLabel, tweet.rawContent, tweet.url, tweet.user.location, str(tweet.inReplyToUser), tweet.retweetedTweet])
                
                    
#                     # Creating a dataframe to load the list
#             tweets_df = pd.DataFrame(attributes_container, columns=["Tweetid", "Tconid", "User", "Name", "Bio", "Created", "Propic","Followers", "Likes", "Retweet", "Listed", "Device", "Tweet", "URL", "Location","Reply", "Trt" ])

#             tweets_df['Created'] = pd.to_datetime(tweets_df['Created'])
#             tweets_df['Date'] = tweets_df['Created'].map(lambda dt: dt.strftime('%d-%m-%Y'))

#             #################FIND KEYWORD/TOPIC##################

#             def find_keyword(text):
#                keyword=re.findall(keywordfind, text, re.I)
#                return keyword
            
#             tweets_df['Keyw'] = tweets_df['Tweet'].apply(lambda x: find_keyword(x.lower()))

#             # tweets_df['Keyword'] = tweets_df['Keyw'].str.replace(r'\b(\w+)(\s+\1)+\b', r'\1')
#             # tweets_df

#             tweets_df['Keyword'] = (tweets_df['Keyw'].apply(lambda x: ','.join(set(x.split(','))) if isinstance(x, str) else x))

#             # tweets_df['Keyword'] = (tweets_df['Keyw'].astype(str).str.split()
#             #                   .apply(lambda x: OrderedDict.fromkeys(x).keys())
#             #                   .str.join(' '))


#             # tweets_df['Keyword']= (tweets_df["Keywords"].str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(''))

#             @st.cache
#             def find_mention(text):
#                 mentname=re.findall(r'(?<![@\w])@(\w{1,25})',text)
#                 return ",".join(mentname)
#             tweets_df['Mentions'] = tweets_df['Tweet'].apply(lambda x: find_mention(x))
#             tweets_df['Splitmentions'] = tweets_df['Mentions'].apply(lambda x: x.split(','))

#             @st.cache
#             def find_ret(text):
#                 rtweet=re.findall(r'^(?=.*\bRT\b\s@\b).*$', text)
#                 return ",".join(rtweet)
#             tweets_df['Retweets']=tweets_df['Tweet'].apply(lambda x: find_ret(x))
#             tweets_df['Splitretweet'] =tweets_df['Retweets'].apply(lambda x: x.split(','))

#             @st.cache
#             def find_hashtag(text):
#                 mentname=re.findall(r'(?<=#)\w+',text)
#                 return ",".join(mentname)
#             tweets_df['Hashtags'] = tweets_df['Tweet'].apply(lambda x: find_hashtag(x))
#             tweets_df['Splithast'] = tweets_df['Hashtags'].apply(lambda x: x.split(','))

#             def remove_usernames_links(tweet):
#                 tweet = re.sub('@[^\s]+','',tweet)
#                 tweet = re.sub('http[^\s]+','',tweet)
#                 return tweet
            
#             tweets_df['Nourl'] = tweets_df['Tweet'].apply(lambda x: remove_usernames_links(x))
            
#             # def removehttp(user):
#             #     user = re.sub('@[^\s]+','',user)
#             #     user = re.sub('http[^\s]+\/','',user)
#             #     return user
#             # tweets_df['Reuser']= tweets_df['Reply'].apply(lambda x: removehttp(x))
#             ######################TRANSLATE#######################

#             translator = Translator()
#             tweets_df['Bioen'] = tweets_df['Bio'].apply(lambda x: translator.translate(x, dest='en').text)

#             tweets_df['Tweeten'] = tweets_df['Nourl'].apply(lambda x: translator.translate(x, dest='en').text)
            

#             ################GENDERPREDICTION#######

#             Classifier  = GndrPrdct()

#             tweets_df['Gender'] = tweets_df['Tweeten'].apply(lambda x: Classifier.predict_gender(x))

#             def map_gender(sex):
#                 if sex == "female":
#                     return "#A459D1"
#                 elif sex == "male":
#                     return "#FFB84C"

#             tweets_df['Colorgend']=tweets_df['Gender'].apply(lambda sex: map_gender(sex)) 

#             ################AGEPREDICTION####################

#             # tweets_df['Ages'] = tweets_df['Tweeten'].apply(lambda x: get_age(x))


#             #################################################
#             spunc=string.punctuation
            
#             def remove_punct(text):
#                 text  = "".join([char for char in text if char not in spunc])
#                 text = re.sub('[0-9]+', '', text)
#                 return text
            
#             tweets_df['Tweetpunct'] = tweets_df['Tweeten'].apply(lambda x: remove_punct(x))

#             def tokenization(text):
#                 text = re.split('\W+', text)
#                 return text

#             tweets_df['Tweettoken'] = tweets_df['Tweetpunct'].apply(lambda x: tokenization(x.lower()))
            
#             stopword = nltk.corpus.stopwords.words('english')
#             def remove_stopwords(text):
#                 text = [word for word in text if word not in stopword]
#                 return text
            
#             tweets_df['Tweetnonstop'] = tweets_df['Tweettoken'].apply(lambda x: remove_stopwords(x))

#             ps = nltk.PorterStemmer()

#             def stemming(text):
#                 text = [ps.stem(word) for word in text]
#                 return text

#             tweets_df['Tweetstem'] = tweets_df['Tweetnonstop'].apply(lambda x: stemming(x))

#             wn = nltk.WordNetLemmatizer()

#             def lemmatizer(text):
#                 text = [wn.lemmatize(word) for word in text]
#                 return text

#             tweets_df['Tweetlem'] = tweets_df['Tweetnonstop'].apply(lambda x: lemmatizer(x))


#             def vader_sentiment(text):
#                 return sentimentweet.polarity_scores(text)['compound']

#             # create new column for vadar compound sentiment score
#             tweets_df['Vdctweet'] = tweets_df['Tweeten'].apply(lambda x: vader_sentiment(x))
#             # tweets_df['sentiment'] = tweets_df['Tweeten'].apply(lambda x: sentiment.polarity_scores(x)['compound'])

#             def catsentiment(sentiment, neg_threshold=-0.05, pos_threshold=0.05):
#                 """ categorise the sentiment value as positive (1), negative (-1) 
#                 or neutral (0) based on given thresholds """
#                 if sentiment < neg_threshold:
#                     label = 'negative'
#                 elif sentiment > pos_threshold:
#                     label = 'positive'
#                 else:
#                     label = 'neutral'
#                 return label

#             # new col with vadar sentiment label based on vadar compound score
#             tweets_df['Vtsenti'] = tweets_df['Vdctweet'].apply(lambda x: catsentiment(x))

            

            

            

#             #############GENSIN#############

#             # def preprocess(df):
#             #     df = clean_text(df)
#             #     df['text'] = \
#             #     df['text'].apply(lambda x: \
#             #     simple_preprocess(x, deacc=True))
#             #     df['text'] = \
#             #     df['text'].apply(lambda x: [word for word in x if \
#             #     word not in stopwords])
#             #     return df

#             # def create_lda_model(id_dict, corpus, num_topics):
#             #     lda_model = LdaModel(corpus=corpus,
#             #     id2word=id_dict,
#             #     num_topics=num_topics,
#             #     random_state=100,
#             #     chunksize=100,
#             #     passes=10)
#             #     return lda_model
            
#             # df = tweets_df('Tweeten')
#             # df = preprocess(df)
#             # texts = df['Tweeten'].values
#             # id_dict = corpora.Dictionary(texts)
#             # corpus = [id_dict.doc2bow(text) for text in texts]
#             # number_topics = 5
#             # lda_model = create_lda_model(id_dict, corpus, number_topics)
#             # ldamodel=pd.DataFrame(lda_model)
                            

            
            
        

            
#             progress_status=st.empty()


#             if tweets_df.empty:
#                 st.write('')
                
#             else:
#                 num_lines = len(tweets_df)/100
#                 my_bar = st.progress(0)

#                 for i in range(100):
#                     progress_status.write("Progress:"+ str(i+1) +" %")
#                     my_bar.progress(i + 1)
#                     time.sleep(num_lines)
#                     progress_status.write('')
#                     my_bar.empty()

#                 tstweets_df= tweets_df.groupby('Date').size().to_frame("Tweets").reset_index()
#                 st.line_chart(data=tstweets_df, x='Date', y='Tweets', width=0, height=0, use_container_width=True)
 

#                 st.dataframe(tweets_df)




#                 #########################################LDASKLEARN##########################

#                 def clean_text(text):
#                     text_lc = "".join([word.lower() for word in text if word not in spunc]) # remove puntuation
#                     text_rc = re.sub('[0-9]+', '', text_lc)
#                     tokens = re.split('\W+', text_rc)    # tokenization
#                     text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
#                     return text

#                 def create_count_vectorizer(documents):
#                     countVectorizer = CountVectorizer(analyzer=clean_text)
#                     countVector = countVectorizer.fit_transform(documents)
#                     return (countVectorizer, countVector)
                
#                 documents = tweets_df['Tweeten']

#                 def create_and_fit_lda(countVector, num_topics):
#                         lda = LDA(n_components=num_topics, n_jobs=-1)
#                         lda.fit(countVector)
#                         return lda
                
#                 def get_most_common_words_for_topics(model, countVectorizer, n_top_words):
#                     words = countVectorizer.get_feature_names_out()
#                     word_dict = {}
#                     for topic_index, topic in enumerate(model.components_):
#                         this_topic_words = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
#                         word_dict[topic_index] = this_topic_words
#                     return word_dict

#                 def print_topic_words(word_dict):
#                     for key in word_dict.keys():
#                         print(f"Topic {key}")
#                         print("\t", word_dict[key])


#                 def get_train_test_sets(df, countVectorizer):
#                     train, test = train_test_split(list(df['text'].values), test_size = 0.2)
#                     train_data =countVectorizer.fit_transform(train)
#                     test_data = countVectorizer.fit_transform(test)
#                     return (train_data, test_data)

#                 def save_model(lda, lda_path, vect, vect_path):
#                     pickle.dump(lda, open(lda_path, 'wb'))
#                     pickle.dump(vect, open(vect_path, 'wb'))

#                 number_topics = 5

#                 # countVectorizer = CountVectorizer(analyzer=clean_text) 
#             # countVector = countVectorizer.fit_transform(tweets_df['Tweeten'])
#             # count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names_out())

#                 (countVectorizer, countVector) = create_count_vectorizer(documents)

#                 lda = create_and_fit_lda(countVector, number_topics)

#                 topic_words = \
#                     get_most_common_words_for_topics(lda, countVectorizer, 10)
#                 print_topic_words(topic_words)

#                 topic_df=pd.DataFrame(topic_words)

#                 save_model(lda, model_path, countVectorizer, vectorizer_path)

                










                

#                 # st.dataframe(count_vect_df)
#                 # st.dataframe(topic_df)
                

                
#                 # @st.cache
#                 # def convert_df(df):
#                 #     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#                 #     return df.to_csv().encode('utf-8')

#                 # csv = convert_df(tweets_df)

#                 # st.download_button(
#                 #     label="Download data as CSV",
#                 #     data=csv,
#                 #     file_name='large_df.csv',
#                 #     mime='text/csv',
#                 # )

#                 st.dataframe(topic_df)


#                 tweets_df.to_pickle('data/dummy.pkl', compression='infer', protocol=4)





                
#                 # print(countVector)
#                 # print(ldaoutput)
            

#         with outer_cols_a:

#             #######################################################
#             G = nx.Graph()

#             for r in tweets_df.iterrows():
#                 G.add_node(r[1]['User'], gender=r[1]['Gender'], color=r[1]['Colorgend'], device=r[1]['Device'], location=r[1]["Location"])
#                 for user in r[1]['Splitmentions']:
#                     G.add_edge(r[1]['User'], user, label='@')
#                 for user in r[1]['Splithast']:
#                     G.add_edge(r[1]['User'], user, weight=2, color='grey', label='#')
#                 for keyword in r[1]['Keyword']:
#                     G.add_edge(r[1]['User'], keyword, color='green', label='topic')

            
                

#             if G.has_node(''):
#                 G.remove_node('')

#             # list(G.edges())[:]
#             # list(G.nodes(data=True))[:]
        
            
#             # color_map=nx.get_node_attributes(G,'gender')
#             # for key in color_map:
#             #     if color_map[key]== 1:
#             #         color_map[key]='blue'
#             #     if color_map[key]== 0:
#             #             color_map[key]='red'

#         #    gender_color=[color_map.get(node) for node in G.nodes()]

        
#             nt=Network('700px', '100%', notebook=True, directed=True, neighborhood_highlight=True, select_menu=True,filter_menu=True, cdn_resources='remote')
#             nt.show_buttons(filter_=['physics'])
        
#         #    # print(neighbor_map)
        

#             nt.from_nx(G)
#             neighbor_map= nt.get_adj_list()
#             for node in nt.nodes:
#                 # print(node['id'])
#                 node['value']=len(neighbor_map[node['id']])
#                 node['title']=' Neighbors:<br>' + '<br>'.join(neighbor_map[node['id']])
#                 # node['color']= color_map[node['id']]

#             # degcent = nx.degree_centrality(G)
#             eigen_vector=nx.eigenvector_centrality(G)
#             # betwen= nx.betweenness_centrality(G)
#             # closeness=nx.closeness_centrality(G)
#             # indeg=nx.in_degree_centrality(G)
#             brid=nx.bridges(G)

#             # nx.set_node_attributes(G, degcent, "centrality")

#             # c = nxcom(G)
#             # # Count the communities
#             # len(c)

#             # try:
#             #     # Find and print node sets
#             #     left, right = bipartite.sets(G)
#             #     print("Left nodes\n", left)
#             #     print("\nRight nodes\n", right)
#             # except NetworkXError as e:
#             #     # Not an affiliation network
#             #     st.write(e)


            
            
#         #    # try:
#         #    # path = '/tmp'
#         #    # nt.save_graph(f'{path}/pyvis_graph.html')
#         #    # HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

#         #    # except:
#             path ='html_files'

#             nt.save_graph(f'{path}/pyvis_graph.html')
#             HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding="utf-8")
            

#             if tweets_df.empty:
#                 st.write("## Scrape tweets by a keyword or some to analyze")
                
#             else:
#                 st.write("## Scraping", maxTweets, " tweets is completed. Here are the analyses:")

#                 components.html(HtmlFile.read(), height=1500, scrolling=False)
#                 # st.write("deg_centrality")
#                 # st.write(deg_centrality)
#                 # st.write("betweeness")
#                 # st.write(betweeness)
#                 # st.write("closeness")
#                 # st.write(closeness)
#                 # st.write("indegree")
#                 # st.write(indegree)

#                 # list(nx.bridges(G))
#                 # st.write(eigen_vector)
                

#                 grouped=tweets_df.groupby(by=['Gender','Vtsenti', 'Vdctweet','User']).count()[["Tweet"]].rename(columns={"Tweet":"Count"})
#                 grouped["Tweets"] = "Tweets"
#                 grouped = grouped.reset_index()
#                 grouped.head()
#                 # st.dataframe(grouped)

                
#                 fig = px.sunburst(grouped,
#                     path=["Tweets", "Gender","Vtsenti", 'Vdctweet','User'],
#                     values='Count',
#                     title="Sentiment Based Gender",
#                     width=700, height=700)
                
#                 st.plotly_chart(fig, use_container_width=True)






    

elif choose == 'Hastags':
     st.write ("BY HASTAGHS")


# #    attributes_container = []
# #    with st.form(key="twithash_form"):
# #       keywords = st_tags(
# #          label='# Enter Keywords:',
# #          text='Press enter to add more',
# #          value=['Zero', 'One', 'Two'],suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'], 
# #          maxtags = 4, key='tweetkeywords')
# #       keywordstr=' '.join(map(str,keywords))
# #       a, b = st.columns([1, 1])
# #       defaultenddate = date.today()
# #       defaultstartdate = defaultenddate - timedelta(days=1)
# #       startdate = a.date_input("Start date", defaultstartdate)
# #       enddate = b.date_input("End date", defaultenddate)
# #       # sdatestr = startdate.strftime("%Y/%m/%d")
# #       # edatestr = enddate.strftime("%Y/%m/%d")
# #       maxTweets = st.number_input('Insert a number', 0)
# #       submit= st.form_submit_button(label="Submit")
# #       if submit:
# #          for i, tweet in enumerate(sntwit.TwitterHashtagScraper( keywordstr + ' since:%s until:%s ' %(startdate, enddate), top = True ).get_items()):
# #             if i>maxTweets:
# #                break
# #          attributes_container.append([tweet.user.username, tweet.user.displayname, tweet.date, tweet.user.profileImageUrl, tweet.user.followersCount, tweet.likeCount, tweet.retweetCount, tweet.user.listedCount, tweet.sourceLabel, tweet.rawContent, tweet.url, tweet.user.location])
# #             # Creating a dataframe to load the list
# #          tweets_df = pd.DataFrame(attributes_container, columns=["User", "Name", "Created", "Propic","Followers", "Likes", "Retweet", "Listed", "Device", "Tweet", "URL", "Location" ])




elif choose == 'Users':
   st.write('Users')
#    useratt_container = []
#    with st.form(key="twituser_form"):
#       Users = st_tags(
#          label='# Enter Keywords:',
#          text='Press enter to add more',
#          value=['Zero', 'One', 'Two'],suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'], 
#          maxtags = 4, key='tweetusers')
#       userstr=' OR '.join(map(str, Users))
#       a, b = st.columns([1, 1])
#       defaultenddate = date.today()
#       defaultstartdate = defaultenddate - timedelta(days=1)
#       startdate = a.date_input("Start date", defaultstartdate)
#       enddate = b.date_input("End date", defaultenddate)
#       # sdatestr = startdate.strftime("%Y/%m/%d")
#       # edatestr = enddate.strftime("%Y/%m/%d")
#       maxTweets = st.number_input('Insert a number', 0)
#       submit= st.form_submit_button(label="Submit")
#       if submit:
#          for i, tweet in enumerate(sntwit.TwitterUserScraper( userstr + ' since:%s until:%s ' %(startdate, enddate), top=True)._get_entity()):
#             if i>maxTweets:
#                break
#         #  useratt_container.append([tweet.user.username, tweet.user.displayname, tweet.date, tweet.user.profileImageUrl, tweet.user.followersCount, tweet.likeCount, tweet.retweetCount, tweet.user.listedCount, tweet.sourceLabel, tweet.rawContent, tweet.url, tweet.user.location])
#         #     # Creating a dataframe to load the list
#         #  tweetuser_df = pd.DataFrame(useratt_container, columns=["User", "Name", "Created", "Propic","Followers", "Likes", "Retweet", "Listed", "Device", "Tweet", "URL", "Location" ])

#         #  st.dataframe (tweetuser_df)
#          print(userstr)
#          print(tweet)
else:
    st.write("File")