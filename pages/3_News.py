import streamlit as st
from newsapi.newsapi_client import NewsApiClient
import json
import pandas as pd
import numpy as np
import datetime
import nltk
from nltk.tree import Tree
import os
import re
import gensim
from nltk import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from googletrans import Translator, constants
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
import folium
import plotly.graph_objs as go
import altair as alt
from altair import datum
import seaborn as sns
import seaborn.objects as so

# from gensim.utils import simple_preprocess
from pprint import pprint
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models
# import PylDavis
import pyLDAvis.gensim
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import utils
from utils import logout
from streamlit_extras.switch_page_button import switch_page


# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
punctuation = (
    "!\"$%&'()*+,-./:;<=>?[\\]^_`{|}~•@"  # define a string of punctuation symbols
)
newsapi = NewsApiClient(api_key='28117c7eb02e41088ba0202989e52b20')
newsdatapath='newsdata'



st.set_page_config(page_title='News Analysis', page_icon=':newspaper:', layout='wide')
st.title('News Analysis')

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div.vis-configuration-wrapper{display: block; width: 400px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

with st. sidebar:
    if st.button("Logout"):
        logout()
        switch_page('home')

with st.sidebar:

        filelist = []
        for root, dirs, files in os.walk("newsdata"):
            for file in files:
                filename = file
                filelist.append(filename)
        # st.write(filelist)

        optionfile = st.selectbox("Select file:", filelist, index=0)

        newsdf = pd.read_json("newsdata/" + optionfile)
        # articledf1['DateTime'] = pd.to_datetime(articledf1['DateTime'], unit='ms')

        st.write("Choosen file to analyze:", optionfile)


outer_cols_a, outer_cols_b = st.columns([4, 2])
# Create form to get user input
with outer_cols_b:
    st.header('Get Data')
    with st.form(key='news_form'):
        keywords = st.text_input(label='Enter keywords to search')
        start_date = st.date_input(label='Choose start date')
        end_date = st.date_input(label='Choose end date')
        submit_button = st.form_submit_button(label='Search')

    # Call News API to get articles based on user input
    filename = ''
    if submit_button:
        filename = f'{keywords}_{start_date}_{end_date}.json'.replace(" ", "_")
        articles = newsapi.get_everything(q=keywords, from_param=start_date, to=end_date)
        
        with open(f"{newsdatapath}/{filename}", 'w') as f:
            json.dump(articles, f)

        # Display article information
        st.write(f'Total Results: {articles["totalResults"]}')
        for article in articles['articles']:
            st.write(f'**Title:** {article["title"]}')
            st.write(f'**Author:** {article["author"]}')
            st.write(f'**Published at:** {article["publishedAt"]}')
            st.write(f'**Description:** {article["description"]}')
            st.write(f'**Source:** {article["source"]["name"]}')
            st.write(f'**URL:** {article["url"]}')
            st.write('---')

    

with outer_cols_a:
    sentimentnews = SentimentIntensityAnalyzer()

# Check if filename exists and is not an empty string
    # if filename and filename.strip() and os.path.isfile(f"{newsdatapath}/{filename}"):
    #     with open(f"{newsdatapath}/{filename}", 'r') as f:
    #         data = json.load(f)
    #     articles = data['articles']
    #     articledf = pd.DataFrame(articles)
    # st.dataframe(articledf)
    st.dataframe(newsdf)

    st.subheader('Extract Named Entity Recognize PERSON and LOCATION')

    articledf = pd.DataFrame(newsdf['articles'].tolist())
    st.dataframe(articledf)


    st.subheader('Data Cleansing, Sentiment Prediction and Tokenize')

    def remove_links(text):
            """Takes a string and removes web links from it"""
            text = re.sub(r"http\S+", "", text)  # remove http links
            text = re.sub(r"bit.ly/\S+", "", text)  # remove bitly links
            text = text.strip("[link]")  # remove [links]
            text = re.sub(r"pic.twitter\S+", "", text)
            return text

    articledf["Nourl"] = articledf["description"].apply(lambda x: remove_links(x))

    def remove_users(text):
            """Takes a string and removes retext and @user information"""
            text = re.sub(
                "(RT\\s@[A-Za-z]+[A-Za-z0-9-_]+)", "", text
            )  # remove re-text
            text = re.sub("(@[A-Za-z]+[A-Za-z0-9-_]+)", "", text)  # remove texted at
            return text

    def remove_hashtags(text):
        """Takes a string and removes any hash tags"""
        text = re.sub("(#[A-Za-z]+[A-Za-z0-9-_]+)", "", text)  # remove hash tags
        return text

    def remove_av(text):
        """Takes a string and removes AUDIO/VIDEO tags or labels"""
        text = re.sub("VIDEO:", "", text)  # remove 'VIDEO:' from start of text
        text = re.sub("AUDIO:", "", text)  # remove 'AUDIO:' from start of text
        return text

    def tokenize(text):
        """Returns tokenized representation of words in lemma form excluding stopwords"""
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if (
                token not in gensim.parsing.preprocessing.STOPWORDS
                and len(token) > 2
            ):  # drops words with less than 3 characters
                result.append(lemmatize(token))
        return result

    def lemmatize(token):
        """Returns lemmatization of a token"""
        return WordNetLemmatizer().lemmatize(token, pos="v")
    
    #########TRANSLATION############

    translator = Translator()

    articledf["engtransl"] = articledf["Nourl"].apply(
            lambda x: translator.translate(x, dest="en").text
        )

    def news_cleansing(text):
        """Main master function to clean texts only without tokenization or removal of stopwords"""
        text = remove_users(text)
        text = remove_links(text)
        text = remove_hashtags(text)
        text = remove_av(text)
        text = text.lower()  # lower case
        text = re.sub('[' + punctuation + ']+', ' ', text)  # strip punctuation
        text = re.sub('\\s+', ' ', text)  # remove double spacing
        text = re.sub('([0-9]+)', '', text)  # remove numbers
        text = re.sub('📝 …', '', text)
        return text
    
    ##############SENTIMENT##########################
    def vader_sentiment(text):
        return sentimentnews.polarity_scores(text)["compound"]

        # create new column for vadar compound sentiment score
    articledf["Compoundscore"] = articledf["engtransl"].apply(
            lambda x: vader_sentiment(x)
        )

    def catsentiment(sentiment, neg_threshold=-0.05, pos_threshold=0.05):
        """categorise the sentiment value as positive (1), negative (-1)
        or neutral (0) based on given thresholds"""
        if sentiment < neg_threshold:
            label = "negative"
        elif sentiment > pos_threshold:
            label = "positive"
        else:
            label = "neutral"
        return label

    # new col with vadar sentiment label based on vadar compound score
    articledf["Sentiment"] = articledf["Compoundscore"].apply(
        lambda x: catsentiment(x)
    )

    def tokenize_news(df):
            """Main function to read in and return cleaned and preprocessed dataframe.
            This can be used in Jupyter notebooks by importing this module and calling the tokenize_tweets() function
            Args:
                df = data frame object to apply cleaning to
            Returns:
                pandas data frame with cleaned tokens
            """

            articledf["tokens"] = articledf.engtransl.apply(news_cleansing)
            num_news = len(articledf)
            print(
                "Complete. Number of Tweets that have been cleaned and tokenized : {}".format(
                    num_news
                )
            )
            return df
    
    articledf = tokenize_news(articledf)

    
    # Define a function to perform NER on a given text

    def extract_entities(text):
    # Tokenize the text into sentences
        sentences = nltk.sent_tokenize(text)

        # Tokenize each sentence into words
        words = [nltk.word_tokenize(sent) for sent in sentences]

        # Perform part-of-speech tagging on each word
        tagged_words = [nltk.pos_tag(w) for w in words]

        # Perform named entity recognition on the tagged words
        chunked_words = [nltk.ne_chunk(w) for w in tagged_words]

        # Extract the PERSON and LOCATION entities from the chunked words
        person_entities = []
        location_entities = []
        for tree in chunked_words:
            for chunk in tree.subtrees():
                if chunk.label() == 'PERSON':
                    person_entities.append(' '.join(c[0] for c in chunk.leaves()))
                elif chunk.label() == 'GPE':
                    location_entities.append(' '.join(c[0] for c in chunk.leaves()))

        return person_entities, location_entities

# Apply the extract_entities function to the 'description' column of the DataFrame
    articledf[['person', 'location']] = articledf['description'].apply(lambda x: pd.Series(extract_entities(x)))
    articledf['person']=articledf['person'].apply(lambda x: "'" + "', '".join(x) + "'")
    articledf['location']=articledf['location'].apply(lambda x: "'" + "', '".join(x) + "'")
    # st.write(articledf.dtypes)
    edited_df=st.experimental_data_editor(articledf )
   
   

    

# Display the updated DataFrame
    st.dataframe(articledf)
    # st.write(articledf.dtypes)
    # editable_df = st.experimental_data_editor(articledf)
    # Convert the DataFrame to an EditableDataFrame
   

    # Display the editable table
    



##################TIMESERIES#####################
    st.subheader("Time Series of News")
        # unique_date = np.unique(date)
    nts = articledf.groupby(by=["publishedAt", "author"]).count()["title"].reset_index()

    factr = plt.figure(figsize=(28, 12))
    ax = sns.scatterplot(
        x="publishedAt", y="title", data=nts, hue="author", palette="deep", s=200
    )
    sns.set(style="whitegrid", font_scale=1)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        ncol=1,
        borderaxespad=0,
        fontsize="27",
    )

    st.pyplot(factr)

    with st.expander("See Data"):
        st.dataframe(nts)



        
#######################  VOICE ##################
    st.header("Voice")
    st.subheader("Topic")

    news = pd.read_json("newsdata/" + optionfile)
    posts = [x.split(" ") for x in articledf["tokens"]]
    # print(posts)

    id2word = corpora.Dictionary(posts)
    corpus_tf = [id2word.doc2bow(text) for text in posts]
    # print(corpus_tf)

    tfidf = models.TfidfModel(corpus_tf)
    corpus_tfidf = tfidf[corpus_tf]
    # print(corpus_tfidf)

    model = LdaModel(
        corpus=corpus_tf, id2word=id2word, alpha=0.1, eta=0.1, random_state=0, num_topics=5
    )
    coherence = CoherenceModel(
        model=model, texts=posts, dictionary=id2word, coherence="u_mass"
    )

    # print(coherence.get_coherence())

    lda_display = pyLDAvis.gensim.prepare(model, corpus_tf, id2word, sort_topics=False)
    pyLDAvis.display(lda_display)

    pyLDAvis.prepared_data_to_html(
        lda_display,
        d3_url=None,
        ldavis_url=None,
        ldavis_css_url=None,
        template_type="general",
        visid=None,
        use_http=True,
    )

    pyLDAvis.save_html(lda_display, "lda.html")
    with open("./lda.html", "r") as f:
        html_string = f.read()
    components.html(html_string, height=800, width=1000, scrolling=True)

    data_dict = {"dominant_topic": [], "perc_contribution": [], "topic_keywords": []}

    for i, row in enumerate(model[corpus_tf]):
        # print(i)
        row = sorted(row, key=lambda x: x[1], reverse=True)
        # print(row)
        for j, (topic_num, prop_topic) in enumerate(row):
            wp = model.show_topic(topic_num)
            topic_keywords = " ".join([word for word, prop in wp])
            data_dict["dominant_topic"].append(int(topic_num))
            data_dict["perc_contribution"].append(round(prop_topic, 3))
            data_dict["topic_keywords"].append(topic_keywords)
            # print(topic_keywords)
            break

    df_topics = pd.DataFrame(data_dict)

    contents = pd.Series(posts)

    df_topics["description"] = articledf["engtransl"]
    df_topics.head()

    ############################
    st.subheader("All of Topics")
    all_topics = {}
    num_terms = 10  # Adjust number of words to represent each topic
    lambd = 1.0  # Adjust this accordingly based on tuning above
    for i in range(
        1, 6
    ):  # Adjust this to reflect number of topics chosen for final LDA model
        topic = lda_display.topic_info[
            lda_display.topic_info.Category == "Topic" + str(i)
        ].copy()
        topic["relevance"] = topic["loglift"] * (1 - lambd) + topic["logprob"] * lambd
        all_topics["Topic " + str(i)] = (
            topic.sort_values(by="relevance", ascending=False).Term[:num_terms].values
        )

    
    alltopic = pd.DataFrame(all_topics).T
    st.dataframe(alltopic, use_container_width=True)

    st.subheader("Topic of News")
    # st.dataframe(df_topics)
    newstopic = news.join(df_topics)
    translator = Translator()
    newstopic["topickeywords_ind"] = newstopic["topic_keywords"].apply(
        lambda x: translator.translate(x, dest="id").text
    )
    st.dataframe(newstopic)

    from wordcloud import WordCloud, ImageColorGenerator

    st.subheader("Wordcloud of Topics")
    wordcloud = WordCloud().generate(" ".join(newstopic["topickeywords_ind"]))

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    with st.expander("See Data"):
        st.dataframe(newstopic, use_container_width=True)
        newstopic.drop_duplicates(inplace=True, subset="description")

    n_s = articledf[["engtransl", "Sentiment"]].groupby("Sentiment").count()[["engtransl"]]

    st.subheader("Sentiment Voice")
    layoutss = go.Layout(legend=dict(yanchor="top", y=1.2, xanchor="left", x=-0.1))
    figs_s = go.Figure(
        data=[go.Pie(labels=n_s.index, values=n_s["engtransl"], hole=0.3)],
        layout=layoutss,
    )
    figs_s.update_layout(
        showlegend=True,
        # height=400,
        # width=400,
        margin={"l": 40, "r": 40, "t": 0, "b": 0},
    )
    figs_s.update_traces(textposition="outside", textinfo="label+percent")
    st.plotly_chart(figs_s, use_container_width=True)







    ################LOCATION###############
    

    # def geocode_location(location):
    #     # function to geocode location string into coordinates
    #     # implementation depends on the geocoding service used
    #     return [latitude, longitude]

    # news_df = pd.read_csv("news_data.csv")
    # news_df["Coordinates"] = news_df["Location"].apply(lambda x: geocode_location(x))

    # m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)

    # for i, row in news_df.iterrows():
    #     folium.Marker(row["Coordinates"], popup=row["Headline"]).add_to(m)

    # st.write(m)
