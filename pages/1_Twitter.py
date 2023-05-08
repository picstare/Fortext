import time
import streamlit as st
import pandas as pd
import numpy as np
import notebook
from datetime import date, timedelta

# from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import re
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px
from PIL import Image
from pyvis import network as net
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy as sp
import streamlit.components.v1 as components

# from streamlit_tags import st_tags
import altair as alt
from altair import datum
import seaborn as sns
import seaborn.objects as so
from tokenisasi import Tokenizer
import functools
from genderpred import GndrPrdct
import math
import sys

# from math import sin
# from nltk.util import ngrams
import string

# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation as LDA
# from sklearn.model_selection import train_test_split
# import pyLDAvis.sklearn
import nltk
from googletrans import Translator, constants

# import streamlit_nested_layout
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from agepred import get_age
from collections import OrderedDict
from networkx.algorithms import bipartite
from networkx import NetworkXError

# from networkx.algorithms.community import greedy_modularity_communities as nxcom
import networkx.algorithms.connectivity as nxcon
from networkx.utils import groups
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community.asyn_fluid import asyn_fluidc as af
from networkx.algorithms.components import weakly_connected_components
from networkx.algorithms import community

# from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import k_clique_communities
from networkx.algorithms import bipartite
import sys
import pickle
import pyLDAvis
import os
from networkx.generators import *
import gensim
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora

# from gensim.utils import simple_preprocess
from pprint import pprint
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models
import pyLDAvis.gensim
from wordcloud import WordCloud, ImageColorGenerator
import snscrape.modules.twitter as sntwit
from nltk import WordNetLemmatizer
import utils
from utils import logout
from streamlit_extras.switch_page_button import switch_page

import matplotlib.colors

# sys.stdin.reconfigure(encoding='utf-8')
# sys.stdout.reconfigure(encoding='utf-8')

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")
punctuation = (
    "!\"$%&'()*+,-./:;<=>?[\\]^_`{|}~•@"  # define a string of punctuation symbols
)

model_path = "sklearn/lda_sklearn.pkl"
vectorizer_path = "sklearn/vectorizer.pkl"

st.set_page_config(page_title="Summary.Picanalytics", layout="wide")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("css/pica2.css")

st.write('<base target="_blank">', unsafe_allow_html=True)

####################LOGOUT####################
with st. sidebar:
    if st.button("Logout"):
        logout()
        switch_page('home')

#################STARTPAGE###################

a, b = st.columns([1, 10])

with a:
    st.text("")
    st.image("img/twitterlkogo.png", width=100)
with b:
    st.title("Twitter Analyzer")

st.write(
    "<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>",
    unsafe_allow_html=True,
)
st.write(
    "<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>",
    unsafe_allow_html=True,
)

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div.vis-configuration-wrapper{display: block; width: 400px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

    ##############################

    # if st.session_state["authentication_status"]:
    #     authenticator.logout("Logout", "sidebar")
    #     st.write(f'Welcome *{st.session_state["name"]}*')
    # # elif st.session_state["authentication_status"] is False:
    # #     st.error('Username/password is incorrect')
    # elif st.session_state["authentication_status"] is None:
    #     # st.warning('Please enter your username and password')
    #     switch_page("Home")

    ###############################

with st.sidebar:

        filelist = []
        for root, dirs, files in os.walk("data"):
            for file in files:
                filename = file
                filelist.append(filename)
        # st.write(filelist)

        optionfile = st.selectbox("Select file:", filelist, index=0)

        tweets_df1 = pd.read_pickle("data/" + optionfile)
        # tweets_df1['DateTime'] = pd.to_datetime(tweets_df1['DateTime'], unit='ms')

        st.write("Choosen file to analyze:", optionfile)


tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])

with tab1:
    

    with st.container():

        col1, col2, col3 = st.columns(3)
        colstat1, colstat2, colstat3 = st.columns(3)

        dres = tweets_df1["Text"].str.extractall(r"(#\S+)")[0].value_counts()
        dresn = dres.rename_axis("Tags").reset_index()
        dresnn = dresn.rename(columns={0: "Freq"}, inplace=False)
        dresnn["Date"] = tweets_df1["Date"]

        dmen = tweets_df1["Text"].str.extractall(r"(?<![@\w])@(\w{1,25})").value_counts()
        dment = dmen.rename_axis("Mentions").reset_index()
        dments = dment.rename(columns={0: "Freq"}, inplace=False)

        ###################################STATISTIC DASAR#####################

        usercount1 = tweets_df1["Username"].nunique()
        tweetcount1 = tweets_df1["Text"].nunique()
        locationcount1 = tweets_df1["Location"].nunique()
        hashtagcount1 = dresnn["Tags"].nunique()
        devicecount1 = tweets_df1["Device"].nunique()
        mentioncount1 = dments["Mentions"].nunique()

        with col1:
            st.write(usercount1, "Users")

        with col2:
            st.write(tweetcount1, "Tweets")

        with col3:
            st.write(hashtagcount1, "Hashtag")

        with colstat1:
            st.write(locationcount1, "Locations")

        with colstat2:
            st.write(devicecount1, "Devices")

        with colstat3:
            st.write(mentioncount1, "Mentions")

        #################### TIME SERIES #####################################

        st.header("Time Series")
        # flc=tweets_df1.groupby('DateTime').size().to_frame("Text").reset_index()
        # sole = alt.Chart(flc).mark_line().encode(x='DateTime', y='Text', color=alt.datum('PSI'))

        # st.line_chart(data=flc, x='DateTime', y='Text', width=0, height=0, use_container_width=True)
        # st.altair_chart(sole, use_container_width=True)

        st.subheader("Time Series of Tweets")

        timevariable = ["Date", "MonthDay", "DayName", "Week", "Month", "Year", "Time"]
        optiontime = st.selectbox("Select time:", timevariable, index=0)

        f, ax = plt.subplots(figsize=(25, 10))
        sns.countplot(x=tweets_df1[optiontime])
        ax.set_ylabel("Tweet")
        for p in ax.patches:
            ax.annotate(
                int(p.get_height()), (p.get_x() + 0.05, p.get_height() + 0.2), fontsize=12
            )

        st.pyplot(f)

        with st.expander("See Data"):
            st.dataframe(tweets_df1)

        #########################TIMESERIESHASHTAG####################

        # st.subheader('Time Series of Hashtags')
        # # unique_date = np.unique(date)

        # fhast=plt.figure(figsize=(28,12))
        # ax = sns.scatterplot(x = 'Date',
        #                y = 'Freq',
        #                data = dresnn,
        #                hue='Tags',
        #                palette='deep',
        #                s=200,
        #                )

        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, borderaxespad=0, fontsize='16')

        # st.pyplot(fhast)

        # with st.expander("See Data"):
        #      st.dataframe(dresnn)

        st.subheader("Time Series of Actors")
        # unique_date = np.unique(date)
        ddlc = tweets_df1.groupby(by=["Date", "Username"]).count()["Text"].reset_index()

        factr = plt.figure(figsize=(28, 12))
        ax = sns.scatterplot(
            x="Date", y="Text", data=ddlc, hue="Username", palette="deep", s=200
        )
        sns.set(style="whitegrid", font_scale=1)
        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            ncol=2,
            borderaxespad=0,
            fontsize="16",
        )

        st.pyplot(factr)

        with st.expander("See Data"):
            st.dataframe(ddlc)


    #######################  VOICE ##################
    st.header("Voice")
    st.subheader("Topic")

    tweets = pd.read_pickle("data/" + optionfile)
    posts = [x.split(" ") for x in tweets["tokens"]]
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
    components.html(html_string, height=800, width=1500, scrolling=False)

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

    df_topics["Tweets"] = tweets["Tweeten"]
    df_topics.head()


    coltop1, coltop2 = st.columns([1, 2])

    with coltop1:
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

        st.subheader("Topic of Tweets")
        # st.dataframe(df_topics)
        tweetopic = tweets.join(df_topics)
        translator = Translator()
        tweetopic["topickeywords_ind"] = tweetopic["topic_keywords"].apply(
            lambda x: translator.translate(x, dest="id").text
        )
        st.dataframe(tweetopic)

    with coltop2:

        st.subheader("Wordcloud of Topics")
        wordcloud = WordCloud().generate(" ".join(tweetopic["topickeywords_ind"]))

        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    with st.expander("See Data"):
        st.dataframe(tweetopic, use_container_width=True)
        tweetopic.drop_duplicates(inplace=True, subset="Tweeten")


    vcol1, vcol2 = st.columns(2)

    with vcol1:

        st.subheader("Top 10 Hashtag Voice")

        # st.dataframe(dresnn)
        toptenhash = dresnn.nlargest(n=10, columns=["Freq"])
        hashtag_chart = (
            alt.Chart(toptenhash)
            .mark_bar()
            .encode(
                x=alt.X("Freq:Q", title="Frequency"),
                y=alt.Y("Tags:N", title="Splithast", sort="-x"),
            )
            .properties(width=700, height=700)
        )

        st.altair_chart(hashtag_chart, use_container_width=False)

    with vcol2:

        st.subheader("Top 10 Mention Voice")

        toptenment = dments.nlargest(n=10, columns=["Freq"])
        mention_chart = (
            alt.Chart(toptenment)
            .mark_bar()
            .encode(
                x=alt.X("Freq:Q", title="Frequency"),
                y=alt.Y("Mentions:N", title="Mentions", sort="-x"),
            )
            .properties(width=700, height=700)
        )

        st.altair_chart(mention_chart, use_container_width=False)

    s_s = tweets_df1[["Tweeten", "Sentiment"]].groupby("Sentiment").count()[["Tweeten"]]
    v_g = tweets_df1[["Tweeten", "Gender"]].groupby("Gender").count()[["Tweeten"]]


    colsenv, colgenv, colsengen = st.columns(3, gap="small")
    with colsenv:
        st.subheader("Sentiment Voice")
        layoutss = go.Layout(legend=dict(yanchor="top", y=1.2, xanchor="left", x=-0.1))
        figs_s = go.Figure(
            data=[go.Pie(labels=s_s.index, values=s_s["Tweeten"], hole=0.3)],
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


    with colgenv:
        st.subheader("Gender Voice")
        layoutvg = go.Layout(legend=dict(yanchor="top", y=1.2, xanchor="left", x=-0.1))
        figv_g = go.Figure(
            data=[go.Pie(labels=v_g.index, values=v_g["Tweeten"], hole=0.3)],
            layout=layoutvg,
        )
        figv_g.update_layout(
            showlegend=True,
            # height=400,
            # width=400,
            margin={"l": 0, "r": 20, "t": 0, "b": 0},
        )
        figv_g.update_traces(textposition="outside", textinfo="label+percent")
        st.plotly_chart(figv_g, use_container_width=True)

    grouped = (
        tweets_df1.groupby(by=["Gender", "Sentiment", "Compoundscore", "Username"])
        .count()[["Text"]]
        .rename(columns={"Text": "Count"})
    )
    grouped["Text"] = "Tweets"
    grouped = grouped.reset_index()
    grouped.head()
    # st.dataframe(grouped)
    with colsengen:
        st.subheader("Sentiment Voice Gender Based ")
        figsemvoice = px.sunburst(
            grouped,
            path=["Text", "Gender", "Sentiment"],
            values="Count",
            # width=500, height=500
        )
        figsemvoice.update_traces(textinfo="label+percent parent")

        st.plotly_chart(figsemvoice, use_container_width=True)
    with st.expander("See Data"):
        st.dataframe(grouped)
    ###################################################################

    ################################ PLACE ############################
    with st.container():
        st.header("Place")
        colp1, colp2, colp3 = st.columns(3)
        with colp1:
            st.subheader("Sentiment Place")
            df_senman = (
                tweets_df1[["Username", "Sentiment"]]
                .groupby("Sentiment")
                .count()[["Username"]]
            )
            layoutsent = go.Layout(
                legend=dict(yanchor="top", y=1.2, xanchor="left", x=-0.2)
            )
            fig_senman = go.Figure(
                data=[
                    go.Pie(labels=df_senman.index, values=df_senman["Username"], hole=0.3)
                ],
                layout=layoutsent,
            )
            fig_senman.update_layout(
                showlegend=True,
                # height=500,
                # width=400,
                margin={"l": 20, "r": 60, "t": 60, "b": 0},
            )
            fig_senman.update_traces(textposition="outside", textinfo="label+percent")
            st.plotly_chart(fig_senman, use_container_width=True)

        with colp2:
            st.subheader("Gender Place")
            df_gender = (
                tweets_df1[["Username", "Gender"]].groupby("Gender").count()[["Username"]]
            )
            layoutgend = go.Layout(
                legend=dict(yanchor="top", y=1.2, xanchor="left", x=-0.1)
            )
            fig_gender = go.Figure(
                data=[
                    go.Pie(labels=df_gender.index, values=df_gender["Username"], hole=0.3)
                ],
                layout=layoutgend,
            )
            fig_gender.update_layout(
                showlegend=True,
                # height=500,
                # width=400,
                margin={"l": 60, "r": 60, "t": 0, "b": 0},
            )
            fig_gender.update_traces(textposition="outside", textinfo="label+percent")
            st.plotly_chart(fig_gender, use_container_width=True)

        with colp3:
            st.subheader("Device Place")

            df_device = (
                tweets_df1[["Username", "Device"]].groupby("Device").count()[["Username"]]
            )
            layoutdev = go.Layout(
                legend=dict(
                    orientation="h",
                    y=-0.8,
                    yanchor="bottom",
                    # xanchor='right',
                    x=0,
                )
            )
            fig_device = go.Figure(
                data=[
                    go.Pie(labels=df_device.index, values=df_device["Username"], hole=0.3)
                ],
                layout=layoutdev,
            )
            fig_device.update_layout(
                showlegend=True,
                height=500,
                width=420,
                margin={"l": 20, "r": 20, "t": 0, "b": 0},
            )
            fig_device.update_traces(textposition="outside", textinfo="label+percent")
            st.plotly_chart(fig_device, use_container_width=False)

        colgeo, colsenp = st.columns([1, 1])
        with colgeo:
            st.subheader("Geo Place")
            df_location = (
                tweets_df1[["Username", "Location"]]
                .groupby("Location")
                .count()[["Username"]]
            )
            layout = go.Layout(legend=dict(orientation="h"))
            fig_location = go.Figure(
                data=[
                    go.Pie(
                        labels=df_location.index, values=df_location["Username"], hole=0.3
                    )
                ]
            )
            fig_location.update_layout(
                showlegend=True,
                height=400,
                width=400,
                margin={"l": 0, "r": 0, "t": 0, "b": 20},
            )
            fig_location.update_traces(textposition="inside", textinfo="label+percent")
            st.plotly_chart(fig_location, use_container_width=True)

        # st.map(tweets_df1['Location'])

        #################################################
        with colsenp:
            st.subheader("Semanticized Place")
            semplace = (
                tweets_df1.groupby(by=["Gender", "Sentiment", "Device", "Location"])
                .count()[["Username"]]
                .rename(columns={"Username": "Count"})
            )
            semplace["Username"] = "User"
            semplace = semplace.reset_index()
            semplace.head()

            figsemplace = px.sunburst(
                semplace,
                path=["Username", "Gender", "Sentiment", "Device", "Location"],
                values="Count",
                width=600,
                height=600,
            )
            figsemplace.update_traces(textinfo="label+percent parent")

            st.plotly_chart(figsemplace, use_container_width=True)
            # with st.expander('See Data'):
            #     st.dataframe(semplace)

    ##############################################


    ####################################### SOCIAL NETWORKX ####################
    st.subheader("Social Network")

    pyvis_network = net.Network(
        "800px",
        "100%",
        directed=True,
        neighborhood_highlight=True,
        select_menu=True,
        filter_menu=True,
        cdn_resources="remote",
    )

    pyvis_network.show_buttons("physics")


    G = nx.DiGraph()

    for r in tweetopic.iterrows():
        G.add_node(
            r[1]["Username"],
            gender=r[1]["Gender"],
            color=r[1]["Colorgend"],
            device=r[1]["Device"],
            location=r[1]["Location"],
            font="20px arial black",
        )
        for user in r[1]["Splitmentions"]:
            G.add_edge(r[1]["Username"], user, label="@")

    H = nx.DiGraph()
    for r in tweetopic.iterrows():
        for user in r[1]["Splithast"]:
            H.add_edge(r[1]["Username"], user, weight=1, color="green", label="#")


    if G.has_node(""):
        G.remove_node("")
    if H.has_node(""):
        H.remove_node("")

    # con=list(nx.connected_components(G))
    # listconnectedcomponent=pd.DataFrame(con)
    # st.dataframe(listconnectedcomponent)


    pyvis_network.from_nx(G)
    # pyvis_network.from_nx(H)
    # pyvis_network.from_nx(T)


    gi1, gi2, gi3 = st.columns(3, gap="large")
    with gi1:
        numnode = G.number_of_nodes()
        st.write("Number of nodes:", numnode)
    with gi2:
        numedge = G.number_of_edges()
        # totedge=G.size()
        st.write("Number of edges:", numedge)
    with gi3:
        selfloop = nx.number_of_selfloops(G)
        st.write("Self loop:", selfloop)
    # with gi4:
    #     connectivity = nx.number_connected_components(G)
    #     st.write("Connected Components:", connectivity)


    colmden, colmcut, colmavc = st.columns(3, gap="large")
    with colmden:
        mden = nx.density(G)
        st.write("Density:", mden)
    with colmcut:
        mcut = nx.average_node_connectivity(G)
        st.write("Connnectivity:", mcut)
    # # with colmeen:
    #     nodes_degree = nx.degree(G)
    #     # display(nodes_degree)
    #     degree_list = []
    #     for (n,d) in nodes_degree:
    #         degree_list.append(d)

    #     av_degree = sum(degree_list) / len(degree_list)

    #     print('The average degree is %s' % av_degree)

    #     def degree_distribution(G):
    #         vk = dict(G.degree())
    #         vk = list(vk.values()) # we get only the degree values
    #         maxk = np.max(vk)
    #         mink = np.min(vk)
    #         kvalues= np.arange(0,maxk+1) # possible values of k
    #         Pk = np.zeros(maxk+1) # P(k)
    #         for k in vk:
    #             Pk[k] = Pk[k] + 1
    #             Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one


    #     def gini(x):
    #         x = [xi for xi in x]
    #         n = len(x)
    #         gini_num = sum([sum([abs(x_i - x_j) for x_j in x]) for x_i in x])
    #         gini_den = 2.0 * n * sum(x)
    #         return gini_num / gini_den

    # megin=gini(nx.eigenvector_centrality(G).values())
    # st.write ('Gini Coefficient:', megin)
    # def entropy(x):
    # # Normalize
    #     total = sum(x)
    #     x = [xi / total for xi in x]
    #     H = sum([-xi * math.log2(xi) for xi in x])
    #     return H
    # meen=entropy(nx.eigenvector_centrality(G).values())
    # st.write ('Network Entropy:', meen)
    with colmavc:
        mavc = nx.average_clustering(G)
        st.write("Average Clustering:", mavc)


    neighbor_map = pyvis_network.get_adj_list()
    for node in pyvis_network.nodes:
        node["value"] = len(neighbor_map[node["id"]])
        node["title"] = " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
        # node['color']=color_map[node['id']]


    bet = nx.betweenness_centrality(G, normalized=True, endpoints=True)
    isinstance(bet, dict)
    nx.set_node_attributes(G, bet, "betweenness")

    degcent = nx.degree_centrality(G)
    isinstance(degcent, dict)
    nx.set_node_attributes(G, degcent, "centrality")

    # eigenvect=nx.eigenvector_centrality(G)
    # isinstance(eigenvect, dict)
    # nx.set_node_attributes(G, eigenvect, "eigenvec")

    closeness = nx.closeness_centrality(G)
    isinstance(closeness, dict)
    nx.set_node_attributes(G, closeness, "closeness")


    # largest_cc = max(nx.connected_components(G), key=len)
    # print(largest_cc)

    # # if H.has_node(''):
    # #     H.remove_node('')

    # # if T.has_node(''):
    # #     T.remove_node('')

    # # list(G.edges())[:]
    # list(G.nodes(data=True))[:]
    # # list(nx.bridges(G))[:]
    # bicomponents_edges = list(nx.biconnected_component_edges(G))
    # # print(nx.number_of_selfloops(G))
    # # G=nx.path_graph(5)
    # # print(nx.average_shortest_path_length(G)

    # bridges= list(nx.bridges(G))
    # print("jembatan", bridges)

    centrality = (
        pd.DataFrame.from_dict(nx.degree_centrality(G), orient="index", columns=["DC"])
        .rename_axis("Name")
        .reset_index()
    )
    toptencen = centrality.nlargest(n=5, columns=["DC"])

    between = (
        pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient="index", columns=["BC"])
        .rename_axis("Name")
        .reset_index()
    )
    toptenbetween = between.nlargest(n=5, columns=["BC"])

    # eigen=(pd.DataFrame.from_dict(nx.eigenvector_centrality(G), orient='index', columns=['EC'])
    #     .rename_axis('Name')
    #     .reset_index())
    # topeigen=eigen.nlargest(n=10, columns=['EC'])

    close = (
        pd.DataFrame.from_dict(nx.closeness_centrality(G), orient="index", columns=["CC"])
        .rename_axis("Name")
        .reset_index()
    )
    toptenclose = close.nlargest(n=5, columns=["CC"])


    path = "html_files"
    pyvis_network.save_graph(f"{path}/graph.html")
    HtmlFile = open(f"{path}/graph.html", "r", encoding="utf-8")
    components.html(HtmlFile.read(), height=960, scrolling=True)

    ########################################

    st.header("Narrative Network Figures")
    cen1, cen2, cen3 = st.columns(3)
    with cen1:

        st.subheader("TOP 5 Main Actors")
        top10cent = (
            alt.Chart(toptencen)
            .mark_bar()
            .encode(x=alt.X("Name:N", sort="-y"), y=alt.Y("DC:Q"), color=alt.Color("DC:Q"))
            .transform_window(
                rank="rank(DC)", sort=[alt.SortField("DC", order="descending")]
            )
            .transform_filter((alt.datum.rank <= 5))
        )

        st.altair_chart(top10cent, use_container_width=True)
        with st.expander("See Data"):
            st.dataframe(toptencen)

    # with cen2:

    #     st.subheader('TOP 10 Hub Actors')
    # top10eigen=alt.Chart(topeigen).mark_bar().encode(
    # x=alt.X('Name:N', sort='-y'),
    # y=alt.Y("EC:Q"),
    # color=alt.Color("EC:Q", scale=alt.Scale(scheme='bluepurple'))
    # ).transform_window(
    #     rank='rank(EC)',
    #     sort=[alt.SortField( 'EC', order= 'descending')]
    # ).transform_filter((alt.datum.rank < 10))

    # st.altair_chart(top10eigen, use_container_width=True)
    # with st.expander("See Data"):
    #     st.dataframe(topeigen)


    # cen3, cen4=st.columns([1,1])
    with cen2:

        st.subheader("TOP 5 Broker Actors")
        # st.markdown('Broker actor is a figure )
        top10between = (
            alt.Chart(toptenbetween)
            .mark_bar()
            .encode(
                x=alt.X("Name:N", sort="-y"),
                y=alt.Y("BC:Q"),
                color=alt.Color("BC:Q", scale=alt.Scale(scheme="yellowgreenblue")),
            )
            .transform_window(
                rank="rank(BC)", sort=[alt.SortField("BC", order="descending")]
            )
            .transform_filter((alt.datum.rank <= 5))
        )

        st.altair_chart(top10between, use_container_width=True)
        with st.expander("See Data"):
            st.dataframe(toptenbetween)

    with cen3:

        st.subheader("TOP 5 Helper Actors")
        top10close = (
            alt.Chart(toptenclose)
            .mark_bar()
            .encode(
                x=alt.X("Name:N", sort="-y"),
                y=alt.Y("CC:Q"),
                color=alt.Color("CC:Q", scale=alt.Scale(scheme="redpurple")),
            )
            .transform_window(
                rank="rank(CC)", sort=[alt.SortField("CC", order="descending")]
            )
            .transform_filter((alt.datum.rank <= 5))
        )

        st.altair_chart(top10close, use_container_width=True)
        with st.expander("See Data"):
            st.dataframe(toptenclose)

    st.header("Influencer")
    followed = tweets_df1.groupby(["Username", "Followers"], as_index=False).size()
    followed.drop("size", axis=1, inplace=True)
    top10followed = followed.nlargest(n=5, columns=["Followers"])

    listed = tweets_df1.groupby(["Username", "Listed"], as_index=False).size()
    listed.drop("size", axis=1, inplace=True)
    top10listed = listed.nlargest(n=5, columns=["Listed"])

    cinf1, cinf2 = st.columns([1, 1])
    with cinf1:
        st.subheader("Top 5 Followed")
        top10follower = (
            alt.Chart(top10followed)
            .mark_bar()
            .encode(
                x=alt.X("Username:N", sort="-y", title="Name"),
                y=alt.Y("Followers:Q"),
                color=alt.Color("Followers:Q"),
            )
            .transform_window(
                rank="rank(Followers)",
                sort=[alt.SortField("Followers", order="descending")],
            )
            .transform_filter((alt.datum.rank <= 5))
        )

        st.altair_chart(top10follower, use_container_width=True)
        with st.expander("See Data"):
            st.dataframe(followed)

    with cinf2:
        st.subheader("Top 5 Listed")
        top10listed = (
            alt.Chart(top10listed)
            .mark_bar()
            .encode(
                x=alt.X("Username:N", sort="-y", title="Name"),
                y=alt.Y("Listed:Q"),
                color=alt.Color("Listed:Q"),
            )
            .transform_window(
                rank="rank(Listed)", sort=[alt.SortField("Listed", order="descending")]
            )
            .transform_filter((alt.datum.rank <= 5))
        )

        st.altair_chart(top10listed, use_container_width=True)
        with st.expander("See Data"):
            st.dataframe(listed)


    ##########################################################
    colbridg, colcomm = st.columns([1, 1], gap="small")


    def display_communities(communities):
        print("we found %s communities" % len(communities))
        # colors = ['red','green','blue','black','orange', 'yellow', 'purple']
        counter = 0
        for community in communities:
            counter += 1
            print("community_%s is:" % counter)
            print(", ".join(community), "\n")
            colors = ["red", "green", "blue", "black", "orange", "yellow", "purple"]
            nx.draw_networkx_labels(G, pos, font_size=5, font_family="sans-serif")
            nx.draw_networkx_nodes(
                G, pos, nodelist=community, node_color=colors.pop(), alpha=0.5
            )
        nx.draw_networkx_edges(G, pos, style="dashed", width=0.3)
        # nx.draw(G, node_color=colors.pop(), alpha=0.5, node_size=100, font_size=4, linewidths=1, with_labels=True)


    with colbridg:
        st.subheader("Weakly connected components")
        st.markdown(
            "Weakly connected components  are a subgraph that is unreachable from other nodes/vertices of a graph or subgraph."
        )

        pos = nx.spring_layout(G)

        wcommunities = weakly_connected_components(G)

        node_groups = []
        for com in wcommunities:
            node_groups.append(list(com))

        color_map = []
        for node in G:
            if node in node_groups[0]:
                color_map.append("orange")
            else:
                color_map.append("red")
        figw = plt.figure(figsize=(5, 5))
        plt.axis("off")
        nx.draw_networkx_nodes(G, pos, node_color=color_map, alpha=0.3, node_size=200)
        nx.draw_networkx_edges(G, pos, style="dashed", width=0.3)
        nx.draw_networkx_labels(G, pos, font_size=5, font_family="sans-serif")
        # nx.draw_networkx_nodes(G, pos, node_size=200, alpha=0.5)
        plt.show()
        st.pyplot(figw)
        with st.expander("See Data"):
            nodegroupdf = pd.DataFrame(node_groups)

            st.dataframe(nodegroupdf)

            p = nx.shortest_path(G)
            pdf = pd.DataFrame(p)
            st.dataframe(pdf)

        # st.markdown ('Bridge an edge of a graph whose deletion increases the graph\'s number of connected components. Equivalently, an edge is a bridge if and only if it is not contained in any cycle.')
        # pos= nx.spring_layout(G)
        # bridges = []
        # for bridge in nx.bridges(G):
        #     bridges.append(bridge)

        # non_bridges = G.edges - bridges
        # print('we have %s bridges in the graph' % len(bridges))
        # # display(bridges)
        # figb= plt.figure(figsize=(5, 5))
        # plt.axis('off')

        # nx.draw(G, node_color='grey', alpha=0.2, node_size=100, font_size=4, linewidths=1, with_labels=True)
        # nx.draw_networkx_labels(G,pos, font_size=5,font_family='sans-serif')
        # nx.draw_networkx_edges(G, pos, edgelist=bridges, style='solid', width=1.5, edge_color='red')
        # nx.draw_networkx_edges(G, pos, edgelist=non_bridges, style='solid', width=0.7, edge_color='black')

        # nx.draw_networkx_nodes(G, pos, node_size=200, alpha=0.5)

        # # display('the red edges are the bridges'
        # st.pyplot(figb)

        # bridgedf=pd.DataFrame(bridges)

        # st.write('the red edges are the bridges')
        # st.write('There are %s bridges in the graph' % len(bridges))

        # st.write('')
        # st.write('')
        # st.write('')
        # st.write('')
        # st.write('')

        # with st.expander("See Data"):
        #     st.dataframe(bridgedf)


    with colcomm:
        st.subheader("Modularity-based Communities")
        st.markdown(
            "Modularity is a measure of the structure of networks or graphs which measures the strength of division of a network into modules (also called groups, clusters or communities). Networks with high modularity have dense connections between the nodes within modules but sparse connections between nodes in different modules."
        )

        pos = nx.spring_layout(G)
        figcomm = plt.figure(figsize=(5, 5))
        plt.axis("off")
        communities = nx.algorithms.community.modularity_max.greedy_modularity_communities(
            G
        )
        display_communities(communities)
        communitieslist = list(communities)
        communitiesdf = pd.DataFrame(communitieslist)

        st.pyplot(figcomm)
        modularity = nx.algorithms.community.quality.modularity(G, communities)
        st.write("The modularity of this communities is: %s" % modularity)

        partition_quality = nx.algorithms.community.quality.partition_quality(
            G, communities
        )
        st.write(
            "The coverage of this communities is: %s \nand the perfomance is: %s"
            % partition_quality
        )

        with st.expander("See Data"):
            st.dataframe(communitiesdf)


    # with colcliq:
    #     st.subheader('Fluid Communities')
    #     st. markdown('A clique is in some sense a stronger version of a community. A set of nodes forms a clique (equivalently, a complete subgraph) if all possible connections between nodes exist.')
    #     pos= nx.spring_layout(G)
    #     figcliq= plt.figure(figsize=(5, 5))
    #     plt.axis('off')


    #     fluid_communities = []
    #     for community in nx.algorithms.community.asyn_fluid.asyn_fluidc(G, 6):
    #         fluid_communities.append(community)

    #     display_communities(fluid_communities)

    #     partition_quality = nx.algorithms.community.quality.partition_quality(G, fluid_communities)
    #     print("The coverage of this communities is: %s \nand the perfomance is: %s" % partition_quality)

    #     # with st.expander("See Data"):

    #     #     st.dataframe(cliquesdf)


    ##############################
    # pr1 = nx.pagerank(G)
    # prdf=pd.DataFrame(pr1)
    # st.dataframe(pr1)


    # with colfluid:
    #     st.subheader('Fluid Communities')
    #     fluid_communities = []
    #     for community in nx.algorithms.community.asyn_fluid.asyn_fluidc(G, 6):
    #         fluid_communities.append(community)

    #     partition_quality = nx.algorithms.community.quality.partition_quality(G, fluid_communities)

    #     nx.draw_networkx_labels(G,pos, font_size=4,font_family='sans-serif')
    #     nx.draw_networkx_nodes(G, pos, nodelist=fluid_communities, node_size=200, node_color=node_color, alpha=0.2)


    # node_color = [(0.5, 0.5, 0.5) for node in G.nodes]
    # for i, node in enumerate(G.nodes()):
    #     if node in max_clique:
    #         node_color[i] = (0.5, 0.5, 0.9)

    # figw= plt.figure(figsize=(10, 10))
    # plt.axis('off')
    # nx.draw_networkx(G, node_color=node_color, pos=nx.spring_layout)
    # # nx.draw(G, node_color=color_map, node_size=100, font_size=5, linewidths=1, with_labels=True)
    # plt.show()
    # st.pyplot(figw)


    # z=nx.Graph()
    # N_nodes=z.number_of_nodes()

    # for r in tweets_df1.iterrows():
    #     z.add_node(r[1]['Username'], gender=r[1]['Gender'], color=r[1]['Colorgend'], device=r[1]['Device'], location=r[1]["Location"], font='20px arial black' )
    #     for user in r[1]['Splitmentions']:
    #         z.add_edge(r[1]['Username'], user)

    # #Computing centrality
    # degCent = nx.degree_centrality(z)

    # #Descending order sorting centrality
    # degCent_sorted=dict(sorted(degCent.items(), key=lambda item: item[1],reverse=True))

    # #Computing betweeness
    # betCent = nx.betweenness_centrality(z, normalized=True, endpoints=True)

    # #Descending order sorting betweeness
    # betCent_sorted=dict(sorted(betCent.items(), key=lambda item: item[1],reverse=True))

    # #Color for regular nodes
    # color_list=N_nodes*['lightsteelblue']

    # #Getting indices on top 10 nodes for each measure
    # N_top=10
    # colors_top_10=['tab:orange','tab:blue','tab:green','lightsteelblue']
    # keys_deg_top=list(degCent_sorted)[0:N_top]
    # keys_bet_top=list(betCent_sorted)[0:N_top]

    # #Computing centrality and betweeness intersection
    # inter_list=list(set(keys_deg_top) & set(keys_bet_top))


    # #Setting up color for nodes
    # for i in inter_list:
    #   color_list[i]=colors_top_10[2]

    # for i in range(N_top):
    #   if keys_deg_top not in inter_list:
    #     color_list[i]=colors_top_10[0]
    #   if keys_bet_top not in inter_list:
    #     color_list[i]=colors_top_10[1]

    # #Draw graph
    # figw= plt.figure(figsize=(10, 10))
    # plt.axis('off')
    # nx.draw(z,pos=nx.circular_layout,with_labels=True,node_color=color_list)

    # #Setting up legend
    # labels=['Top 10 deg cent','Top 10 bet cent','Top 10 deg and bet cent','no top 10']
    # for i in range(len(labels)):
    #   plt.scatter([],[],label=labels[i],color=colors_top_10[i])
    # plt.legend(loc='center')
    # plt.show()
    # st.pyplot(figw)


    ######################### GAMBAR TOP TEN NETWORK  ###############
    # N_nodes=G.number_of_nodes()
    # G=nx.Graph()
    # fig, ax = plt.subplots()
    # #Computing centrality
    # degCent = nx.degree_centrality(G)

    # #Descending order sorting centrality
    # degCent_sorted=dict(sorted(degCent.items(), key=lambda item: item[1],reverse=True))

    # #Computing betweeness
    # betCent = nx.betweenness_centrality(G, normalized=True, endpoints=True)

    # #Descending order sorting betweeness
    # betCent_sorted=dict(sorted(betCent.items(), key=lambda item: item[1],reverse=True))

    # #Color for regular nodes
    # color_list=N_nodes*['lightsteelblue']

    # #Getting indices on top 10 nodes for each measure
    # N_top=10
    # colors_top_10=['tab:orange','tab:blue','tab:green','lightsteelblue']
    # keys_deg_top=list(degCent_sorted)[0:N_top]
    # keys_bet_top=list(betCent_sorted)[0:N_top]

    # #Computing centrality and betweeness intersection
    # inter_list=list(set(keys_deg_top) & set(keys_bet_top))
    # figz, ax= plt.subplot()
    # #Setting up color for nodes
    # for i in inter_list:
    #   color_list[i]=colors_top_10[2]

    # for i in range(N_top):
    #   if keys_deg_top[i] not in inter_list:
    #     color_list[keys_deg_top[i]]=colors_top_10[0]
    #   if keys_bet_top[i] not in inter_list:
    #     color_list[keys_bet_top[i]]=colors_top_10[1]

    # #Draw graph

    # pos= nx.circular_layout(G)
    # nx.draw(G,pos,with_labels=True,node_color=color_list)

    # #Setting up legend
    # labels=['Top 10 deg cent','Top 10 bet cent','Top 10 deg and bet cent','no top 10']
    # for i in range(len(labels)):
    #   plt.scatter([],[],label=labels[i],color=colors_top_10[i])
    # plt.legend(loc='center')
    # plt.show(fig)

with tab2:
    with st.container():
        attributes_container = []
        outer_cols_a, outer_cols_b = st.columns([4, 2])
        sentimentweet = SentimentIntensityAnalyzer()

        with outer_cols_b:
            with st.form(key="tkeywordform"):
                keywords = st.text_input("Keywords", "")
                # keywords = st_tags(
                #     label='# Enter Keywords:',
                #     text='Press enter to add more',
                #     value=['Zero', 'One', 'Two'],suggestions=['five', 'six', 'seven', 'eight', 'nine', 'three', 'eleven', 'ten', 'four'],
                #     maxtags = 4, key='tweetkeywords')

                username = st.text_input("User Name", "")
                # keywordstr=' OR '.join(map(str,keywords))
                # keywordfind=' | '.join(map(str,keywords))
                defaultenddate = date.today()
                defaultstartdate = defaultenddate - timedelta(days=7)

                inner_cols_a, inner_cols_b = st.columns([1, 1])
                since = inner_cols_a.date_input("Start date", defaultstartdate)
                until = inner_cols_b.date_input("End date", defaultenddate)
                # sdatestr = startdate.strftime("YYYY-MM-DD")
                # edatestr = enddate.strftime("YYYY-MM-DD")

                inner_cols_c, inner_cols_d, inner_cols_e = st.columns(3)
                maxTweets = inner_cols_c.number_input("Insert a number", 0)
                retweets = inner_cols_d.checkbox("Exclude retweets", False)
                replies = inner_cols_e.checkbox("Exclude replies", False)

                submit = st.form_submit_button(label="Submit")

                def search(keywords, username, since, until, retweets, replies):
                    global filename
                    q = keywords
                    if username != "":
                        q += f" from:{username}"
                    if since != "":
                        q += f" since:{since}"
                    if until != "":
                        q += f" until:{until}"
                    if retweets == True:
                        q += f" exclude:retweet"
                    if replies == True:
                        q += f" exclude:replies"
                    if username != "" and keywords != "":
                        filename = f"{since}_{until}_{username}_{keywords}.pkl"
                    elif username != "":
                        filename = f"{since}_{until}_{username}.pkl"
                    else:
                        filename = f"{since}_{until}_{keywords}.pkl"
                    # print(filename)
                    return q

                q = search(keywords, username, since, until, retweets, replies)

                if submit:
                    for i, tweet in enumerate(
                        sntwit.TwitterSearchScraper(q, top=True).get_items()
                    ):
                        if i >= maxTweets:  # number of tweets you want to scrape
                            break
                        attributes_container.append(
                            [
                                tweet.date,
                                tweet.id,
                                tweet.rawContent,
                                tweet.user.username,
                                tweet.user.profileImageUrl,
                                tweet.user.followersCount,
                                tweet.user.listedCount,
                                tweet.user.location,
                                tweet.sourceLabel,
                                tweet.lang,
                                tweet.replyCount,
                                tweet.retweetCount,
                                tweet.likeCount,
                                tweet.quoteCount,
                                tweet.media,
                            ]
                        )

            tweets_df = pd.DataFrame(
                attributes_container,
                columns=[
                    "DateTime",
                    "TweetId",
                    "Text",
                    "Username",
                    "ProfileImageUrl",
                    "Followers",
                    "Listed",
                    "Location",
                    "Device",
                    "Language",
                    "ReplyCount",
                    "RetweetCount",
                    "LikeCount",
                    "QuoteCount",
                    "Media",
                ],
            )

            # st.write(q)

            tweets_df.sort_values(by="DateTime", ascending=False)

            def find_ret(text):
                rtweet = re.findall(r"^(?=.*\bRT\b\s@\b).*$", text)
                return ",".join(rtweet)

            tweets_df["Retweets"] = tweets_df["Text"].apply(lambda x: find_ret(x))
            

            def find_mention(text):
                mentname = re.findall(r"(?<![@\w])@(\w{1,25})", text)
                return ",".join(mentname)

            def find_hashtag(text):
                mentname = re.findall(r"(?<=#)\w+", text)
                return ",".join(mentname)

            tweets_df["Hashtags"] = tweets_df["Text"].apply(lambda x: find_hashtag(x))
            tweets_df["Splithast"] = tweets_df["Hashtags"].apply(lambda x: x.split(","))

            tweets_df["Mentions"] = tweets_df["Text"].apply(lambda x: find_mention(x))
            tweets_df["Splitmentions"] = tweets_df["Mentions"].apply(lambda x: x.split(","))

            ##########################
            # tweets_df['Splithashtags'] = tweets_df['Hashtags'].apply(lambda x: x.split(','))

            ##########################

            def remove_links(tweet):
                """Takes a string and removes web links from it"""
                tweet = re.sub(r"http\S+", "", tweet)  # remove http links
                tweet = re.sub(r"bit.ly/\S+", "", tweet)  # remove bitly links
                tweet = tweet.strip("[link]")  # remove [links]
                tweet = re.sub(r"pic.twitter\S+", "", tweet)
                return tweet

            tweets_df["Nourl"] = tweets_df["Text"].apply(lambda x: remove_links(x))

            #########TRANSLATION############

            translator = Translator()

            tweets_df["Tweeten"] = tweets_df["Nourl"].apply(
                lambda x: translator.translate(x, dest="en").text
            )

            #############SENTIMENT#########################

            def vader_sentiment(text):
                return sentimentweet.polarity_scores(text)["compound"]

            # create new column for vadar compound sentiment score
            tweets_df["Compoundscore"] = tweets_df["Tweeten"].apply(
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
            tweets_df["Sentiment"] = tweets_df["Compoundscore"].apply(
                lambda x: catsentiment(x)
            )

            ################GENDERPREDICTION#######

            Classifier = GndrPrdct()

            tweets_df["Gender"] = tweets_df["Tweeten"].apply(
                lambda x: Classifier.predict_gender(x)
            )

            def map_gender(sex):
                if sex == "female":
                    return "#A459D1"
                elif sex == "male":
                    return "#FFB84C"

            tweets_df["Colorgend"] = tweets_df["Gender"].apply(lambda sex: map_gender(sex))

            ################AGEPREDICTION####################

            # tweets_df['Ages'] = tweets_df['Text'].apply(lambda x: agepred.get_age(x))

            #################################################

            def remove_users(tweet):
                """Takes a string and removes retweet and @user information"""
                tweet = re.sub(
                    "(RT\\s@[A-Za-z]+[A-Za-z0-9-_]+)", "", tweet
                )  # remove re-tweet
                tweet = re.sub("(@[A-Za-z]+[A-Za-z0-9-_]+)", "", tweet)  # remove tweeted at
                return tweet

            def remove_hashtags(tweet):
                """Takes a string and removes any hash tags"""
                tweet = re.sub("(#[A-Za-z]+[A-Za-z0-9-_]+)", "", tweet)  # remove hash tags
                return tweet

            def remove_av(tweet):
                """Takes a string and removes AUDIO/VIDEO tags or labels"""
                tweet = re.sub("VIDEO:", "", tweet)  # remove 'VIDEO:' from start of tweet
                tweet = re.sub("AUDIO:", "", tweet)  # remove 'AUDIO:' from start of tweet
                return tweet

            def tokenize(tweet):
                """Returns tokenized representation of words in lemma form excluding stopwords"""
                result = []
                for token in gensim.utils.simple_preprocess(tweet):
                    if (
                        token not in gensim.parsing.preprocessing.STOPWORDS
                        and len(token) > 2
                    ):  # drops words with less than 3 characters
                        result.append(lemmatize(token))
                return result

            def lemmatize(token):
                """Returns lemmatization of a token"""
                return WordNetLemmatizer().lemmatize(token, pos="v")

            def preprocess_tweet(tweet):
                """Main master function to clean tweets, stripping noisy characters, and tokenizing use lemmatization"""
                tweet = remove_users(tweet)
                tweet = remove_hashtags(tweet)
                tweet = remove_av(tweet)
                tweet = tweet.lower()  # lower case
                tweet = re.sub("[" + punctuation + "]+", " ", tweet)  # strip punctuation
                tweet = re.sub("\\s\\s+", " ", tweet)  # remove double spacing
                tweet = re.sub("([0-9]+)", "", tweet)  # remove numbers
                tweet_token_list = tokenize(tweet)  # apply lemmatization and tokenization
                tweet = " ".join(tweet_token_list)
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

                df["tokens"] = tweets_df.Tweeten.apply(preprocess_tweet)
                num_tweets = len(df)
                print(
                    "Complete. Number of Tweets that have been cleaned and tokenized : {}".format(
                        num_tweets
                    )
                )
                return df

            tweets_df = tokenize_tweets(tweets_df)

            tweets_df["DateTime"] = pd.to_datetime(tweets_df["DateTime"], unit="ms")

            tweets_df["Hour"] = tweets_df["DateTime"].dt.hour
            tweets_df["Year"] = tweets_df["DateTime"].dt.year
            tweets_df["Month"] = tweets_df["DateTime"].dt.month
            tweets_df["MonthName"] = tweets_df["DateTime"].dt.month_name()
            tweets_df["MonthDay"] = tweets_df["DateTime"].dt.day
            tweets_df["DayName"] = tweets_df["DateTime"].dt.day_name()
            tweets_df["Week"] = tweets_df["DateTime"].dt.isocalendar().week

            tweets_df["Date"] = [d.date() for d in tweets_df["DateTime"]]
            tweets_df["Time"] = [d.time() for d in tweets_df["DateTime"]]

            tweets_df.drop("DateTime", axis=1, inplace=True)
            tweets_df.drop("Retweets", axis=1, inplace=True)
            tweets_df.drop("Mentions", axis=1, inplace=True)
            # tweets_df.drop('Mentions',axis=1,inplace=True)
            tweets_df.drop("Hashtags", axis=1, inplace=True)

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
            st.write("")
        else:
            tweets_df.info()
            st.dataframe(tweets_df)
            tweets_df.to_pickle(f"{datapath}/{filename}", compression="infer", protocol=4)