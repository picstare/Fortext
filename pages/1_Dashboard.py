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
from  PIL import Image
from pyvis import network as net
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy as sp
import streamlit.components.v1 as components
# from streamlit_tags import st_tags
import altair as alt
from altair import datum
import seaborn as sns
from tokenisasi import Tokenizer
import functools
from genderpred import GndrPrdct
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
from networkx.utils import groups
# from networkx.algorithms.community.asyn_fluid import asyn_fluidc as af
# from networkx.algorithms.components import weakly_connected_components
import sys
import pickle
import pyLDAvis
import os
from networkx.generators import *
import gensim
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
# from gensim.utils import simple_preprocess
# from pprint import pprint
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models
import pyLDAvis.gensim
from wordcloud import WordCloud, ImageColorGenerator
from Home import authenticator, authentication_status,name
import yaml
from yaml import SafeLoader
from streamlit_authenticator import Authenticate, authenticate
from streamlit_extras.switch_page_button import switch_page

# sys.stdin.reconfigure(encoding='utf-8')
# sys.stdout.reconfigure(encoding='utf-8')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

model_path = "sklearn/lda_sklearn.pkl"
vectorizer_path = "sklearn/vectorizer.pkl"

st.set_page_config(
    page_title="Summary.Picanalytics",
    layout="wide"
      
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("css/pica2.css")

st.write('<base target="_blank">', unsafe_allow_html=True)

a, b = st.columns([1, 10])

with a:
    st.text("")
    st.image("img/twitterlkogo.png", width=100)
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

##############################

# if authentication_status:
#         authenticator.logout('Logout', 'sidebar')
#         # st.write(f'Welcome *{name}*')
#         # st.title('Some content')
# elif authentication_status is False:
#         st.error('Username/password is incorrect')

# if authentication_status is None:
#          switch_page('Home')




###############################

with st.sidebar:
    
    filelist=[]
    for root, dirs, files in os.walk("data"):
        for file in files:
                filename=file
                filelist.append(filename)
    # st.write(filelist)

    optionfile=st.selectbox("Select file:", filelist, index=0)

    tweets_df1 = pd.read_pickle('data/'+optionfile)

    st.write("Choosen file to analyze:", optionfile)
   

    logoutbutton=st.sidebar.button('Logout')
    if logoutbutton:
        # elf.cookie_manager.delete(self.cookie_name)
        st.session_state['logout'] = True
        switch_page('Home')

    
    # authenticator.logout('Logout', 'sidebar')

   
   

with st.container():
    
    col1, col2, col3 =st.columns(3)
    colstat1, colstat2, colstat3=st.columns(3)

    dres=tweets_df1['Text'].str.extractall(r'(\#\w*)')[0].value_counts()
    dresn = dres.rename_axis('Tags').reset_index()
    dresnn=dresn.rename(columns={ 0:'Freq'}, inplace=False )

    dmen=tweets_df1['Text'].str.extractall(r'(?<![@\w])@(\w{1,25})').value_counts()
    dment = dmen.rename_axis('Mentions').reset_index()
    dments=dment.rename(columns={0:'Freq'}, inplace=False)
    
###################################STATISTIC DASAR#####################  

    usercount1= tweets_df1['Username'].nunique()
    tweetcount1=tweets_df1['Text'].nunique()
    locationcount1=tweets_df1['Location'].nunique()
    hashtagcount1=dresnn['Tags'].nunique()
    devicecount1=tweets_df1['Device'].nunique()
    mentioncount1=dments['Mentions'].nunique()
        
    
    with col1:
        st.write(usercount1, 'Users')
        
    with col2:
        st.write(tweetcount1, 'Tweets')
        
    with col3:
        st.write (hashtagcount1, 'Hashtags')
        
    

    with colstat1:
        st.write(locationcount1, 'Locations')

    with colstat2:
        st.write (devicecount1, 'Devices')
    
    with colstat3:
        st.write (mentioncount1, "Mentions")


#################### TIME SERIES #####################################

    st.subheader('Time Series')

    timevariable=['MonthDay', 'DayName', 'Week,' 'Month','Year','Time']
    optiontime=st.selectbox("Select time:", timevariable, index=0)

    f, ax = plt.subplots(figsize=(25, 10))
    sns.countplot(x= tweets_df1[optiontime])
    ax.set_ylabel('Tweet')
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x()+0.05, p.get_height()+0.2), fontsize = 12)

    st.pyplot(f)

###############################################################
    with st.expander("See Data"):
        st.dataframe(tweets_df1)
########################################################################

    
####################################### NETWORKX ####################
st.subheader("Graph Network")

pyvis_network=net.Network('800px', '100%', directed=True, neighborhood_highlight=True, select_menu=True,filter_menu=True, cdn_resources='remote')

pyvis_network.show_buttons('physics')


G = nx.Graph()

for r in tweets_df1.iterrows():
    G.add_node(r[1]['Username'], gender=r[1]['Gender'], color=r[1]['Colorgend'], device=r[1]['Device'], location=r[1]["Location"], font='20px arial black' )
    for user in r[1]['Splitmentions']:
        G.add_edge(r[1]['Username'], user)

if G.has_node(""):
    G.remove_node("")

pyvis_network.from_nx(G)


# communities = weakly_connected_components(G)

# node_groups = []
# for com in communities:
#   node_groups.append(list(com))


# color_map = []
# for node in G:
#     if node in node_groups[0]:
#         color_map.append('orange')
#     else: 
#         color_map.append('red') 
        

# color=color_map

# nx.set_node_attributes(G, color, "color")
# pyvis_network.add_node(node, label=node, color="color")

neighbor_map= pyvis_network.get_adj_list()
for node in pyvis_network.nodes:
        node['value']=len(neighbor_map[node['id']])
        node['title']=' Neighbors:<br>' + '<br>'.join(neighbor_map[node['id']])
        # node['color']=color_map[node['id']]



bet= nx.betweenness_centrality(G, normalized=True, endpoints=True)
isinstance(bet, dict)
nx.set_node_attributes(G, bet, "betweenness")

degcent = nx.degree_centrality(G)
isinstance(degcent, dict)
nx.set_node_attributes(G, degcent, "centrality")

eigenvect=nx.eigenvector_centrality(G)
isinstance(eigenvect, dict)
nx.set_node_attributes(G, eigenvect, "eigenvec")

closeness=nx.closeness_centrality(G)
isinstance(closeness, dict)
nx.set_node_attributes(G, closeness, "closeness")
            
if G.has_node(""):
    G.remove_node("")

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
# # print (nx.density(G))
# # G=nx.path_graph(5)
# # print(nx.average_shortest_path_length(G)

centrality=(pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index', columns=['DC'])
    .rename_axis('Name')
    .reset_index())
toptencen=centrality.nlargest(n=10, columns=['DC'])

between=(pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient='index', columns=['BC'])
    .rename_axis('Name')
    .reset_index())
toptenbetween=between.nlargest(n=10, columns=['BC'])

eigen=(pd.DataFrame.from_dict(nx.eigenvector_centrality(G), orient='index', columns=['EC'])
    .rename_axis('Name')
    .reset_index())
topeigen=eigen.nlargest(n=10, columns=['EC'])

close=(pd.DataFrame.from_dict(nx.closeness_centrality(G), orient='index', columns=['CC'])
    .rename_axis('Name')
    .reset_index())
toptenclose=close.nlargest(n=10, columns=['CC'])


path ='html_files'
pyvis_network.save_graph(f'{path}/graph.html')
HtmlFile = open(f'{path}/graph.html', 'r', encoding="utf-8")
components.html(HtmlFile.read(), height=960, scrolling=True )

########################################

st.header('Narrative Network Figures')
cen1, cen2 = st.columns([1,2])
with cen1:
   
    st.subheader("TOP 10 Main Actors")
    top10cent=alt.Chart(toptencen).mark_bar().encode(
    x=alt.X('Name:N', sort='-y'),
    y=alt.Y("DC:Q"),
    color=alt.Color("DC:Q")
    ).transform_window(
        rank='rank(DC)',
        sort=[alt.SortField( 'DC', order= 'descending')]
    ).transform_filter((alt.datum.rank < 10))

    st.altair_chart(top10cent, use_container_width=True)
    with st.expander("See Data"):
        st.dataframe(toptencen)

with cen2:
    
    st.subheader('TOP 10 Hub Actors')
    top10eigen=alt.Chart(topeigen).mark_bar().encode(
    x=alt.X('Name:N', sort='-y'),
    y=alt.Y("EC:Q"),
    color=alt.Color("EC:Q", scale=alt.Scale(scheme='bluepurple'))
    ).transform_window(
        rank='rank(EC)',
        sort=[alt.SortField( 'EC', order= 'descending')]
    ).transform_filter((alt.datum.rank < 10))

    st.altair_chart(top10eigen, use_container_width=True)
    with st.expander("See Data"):
        st.dataframe(topeigen)


cen3, cen4=st.columns([1,1])
with cen3:
    
    st.subheader('TOP 10 Broker Actors')
    top10between=alt.Chart(toptenbetween).mark_bar().encode(
    x=alt.X('Name:N', sort='-y'),
    y=alt.Y("BC:Q"),
    color=alt.Color("BC:Q", scale=alt.Scale(scheme='yellowgreenblue'))
    ).transform_window(
        rank='rank(BC)',
        sort=[alt.SortField( 'BC', order= 'descending')]
    ).transform_filter((alt.datum.rank <= 10))

    st.altair_chart(top10between, use_container_width=True)
    with st.expander("See Data"):
        st.dataframe(toptenbetween)

with cen4:
    
    st.subheader('TOP 10 Helper Actors')
    top10close=alt.Chart(toptenclose).mark_bar().encode(
    x=alt.X('Name:N', sort='-y'),
    y=alt.Y("CC:Q"),
    color=alt.Color("CC:Q", scale=alt.Scale(scheme='redpurple'))
    ).transform_window(
        rank='rank(CC)',
        sort=[alt.SortField( 'CC', order= 'descending')]
    ).transform_filter((alt.datum.rank <= 10))

    st.altair_chart(top10close, use_container_width=True)
    with st.expander("See Data"):
        st.dataframe(toptenclose)

st.header('Influencer')
followed=tweets_df1.groupby(['Username', 'Followers'], as_index=False).size()
followed.drop('size',axis=1,inplace=True)
top10followed=followed.nlargest(n=10, columns=['Followers'])

listed=tweets_df1.groupby(['Username', 'Listed'], as_index=False).size()
listed.drop('size',axis=1,inplace=True)
top10listed=listed.nlargest(n=10, columns=['Listed'])

cinf1, cinf2=st.columns([1,1])
with cinf1:
    st.subheader('Top 10 Followed')
    top10follower=alt.Chart(top10followed).mark_bar().encode(
    x=alt.X('Username:N', sort='-y', title='Name'),
        y=alt.Y("Followers:Q"),
        color=alt.Color("Followers:Q")
        ).transform_window(
            rank='rank(Followers)',
            sort=[alt.SortField( 'Followers', order= 'descending')]
        ).transform_filter((alt.datum.rank <= 10))

    st.altair_chart(top10follower, use_container_width=True)
    with st.expander("See Data"):
            st.dataframe(followed)

with cinf2:
    st.subheader('Top 10 Listed')
    top10listed=alt.Chart(top10listed).mark_bar().encode(
    x=alt.X('Username:N', sort='-y', title='Name'),
        y=alt.Y("Listed:Q"),
        color=alt.Color("Listed:Q")
        ).transform_window(
            rank='rank(Listed)',
            sort=[alt.SortField( 'Listed', order= 'descending')]
        ).transform_filter((alt.datum.rank <= 10))

    st.altair_chart(top10listed, use_container_width=True)
    with st.expander("See Data"):
            st.dataframe(listed)

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


#########################################
st.header('Voice')
st.subheader('Topic')

tweets=pd.read_pickle('data/'+optionfile)
posts = [x.split(' ') for x in tweets['tokens']]
# print(posts)

id2word = corpora.Dictionary(posts)
corpus_tf = [id2word.doc2bow(text) for text in posts]
# print(corpus_tf)

tfidf = models.TfidfModel(corpus_tf)
corpus_tfidf = tfidf[corpus_tf]
# print(corpus_tfidf)

model = LdaModel(corpus=corpus_tf,id2word = id2word, alpha=.1, eta=0.1, random_state = 0, num_topics = 5)
coherence = CoherenceModel(model = model, texts = posts, dictionary = id2word, coherence = 'u_mass')

# print(coherence.get_coherence())

lda_display = pyLDAvis.gensim.prepare(model, corpus_tf, id2word, sort_topics = False)
pyLDAvis.display(lda_display)

pyLDAvis.prepared_data_to_html(lda_display, d3_url=None, ldavis_url=None, ldavis_css_url=None, template_type='general', visid=None, use_http=True)

pyLDAvis.save_html(lda_display, "lda.html")
with open('./lda.html', 'r') as f:
    html_string = f.read()
components.html(html_string, height=800, width=1500, scrolling=False)

data_dict = {'dominant_topic':[], 'perc_contribution':[], 'topic_keywords':[]}

for i, row in enumerate(model[corpus_tf]):
    #print(i)
    row = sorted(row, key=lambda x: x[1], reverse=True)
    #print(row)
    for j, (topic_num, prop_topic) in enumerate(row):
        wp = model.show_topic(topic_num)
        topic_keywords = " ".join([word for word, prop in wp])
        data_dict['dominant_topic'].append(int(topic_num))
        data_dict['perc_contribution'].append(round(prop_topic, 3))
        data_dict['topic_keywords'].append(topic_keywords)
        #print(topic_keywords)
        break

df_topics = pd.DataFrame(data_dict)

contents = pd.Series(posts)

df_topics['Tweets'] = tweets['Tweeten']
df_topics.head()


coltop1, coltop2=st.columns([1,2])

with coltop1:
    st.subheader('All of Topics')
    all_topics = {}
    num_terms = 10 # Adjust number of words to represent each topic
    lambd = 1.0 # Adjust this accordingly based on tuning above
    for i in range(1,6): #Adjust this to reflect number of topics chosen for final LDA model
            topic = lda_display.topic_info[lda_display.topic_info.Category == 'Topic'+str(i)].copy()
            topic['relevance'] = topic['loglift']*(1-lambd)+topic['logprob']*lambd
            all_topics['Topic '+str(i)] = topic.sort_values(by='relevance', ascending=False).Term[:num_terms].values

    alltopic=pd.DataFrame(all_topics).T
    st.dataframe(alltopic, use_container_width=True)

    st.subheader('Topic of Tweets')
    st.dataframe(df_topics)
    tweetopic=tweets.join(df_topics)

with coltop2:
    

    st.subheader('Wordcloud of Topics')
    wordcloud = WordCloud().generate(' '.join (tweetopic['topic_keywords']))

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)




with st.expander("See Data"):
    st.dataframe(tweetopic, use_container_width=True)
    tweetopic.drop_duplicates(inplace=True, subset='Tweeten')


vcol1, vcol2=st.columns(2)

with vcol1:

    st.subheader('Hashtag Voice')
    
    # st.dataframe(dresnn)
    toptenhash=dresnn.nlargest(n=10, columns=['Freq'])
    hashtag_chart=alt.Chart(toptenhash).mark_bar().encode(
        x=alt.X('Freq:Q', title='Frequency'),
        y=alt.Y("Tags:N", title='Hashtags', sort="-x")
    ).properties(width= 700, height=700)

    st.altair_chart(hashtag_chart, use_container_width=False)

with vcol2:

    st.subheader('Mention Voice')


    toptenment=dments.nlargest(n=10, columns=['Freq'])
    mention_chart=alt.Chart(toptenment).mark_bar().encode(
        x=alt.X('Freq:Q', title='Frequency'),
        y=alt.Y("Mentions:N", title='Mentions', sort="-x")
    ).properties(width= 700, height=700)

    st.altair_chart(mention_chart, use_container_width=False)

s_s = tweets_df1[['Tweeten', 'Sentiment']].groupby('Sentiment').count()[["Tweeten"]]
v_g = tweets_df1[['Tweeten', 'Gender']].groupby('Gender').count()[["Tweeten"]]

colsenv, colgenv, colsengen=st.columns(3)
with colsenv:
    st.subheader('Sentiment Voice')
    layoutss = go.Layout(
                        legend=dict(
                        yanchor="top",
            y=1.2,
            xanchor="left",
            x=-0.1)
                        )
    figs_s = go.Figure(data=[go.Pie(labels=s_s.index,
                                                values=s_s['Tweeten'],
                                                hole=.3)], layout=layoutss)
    figs_s.update_layout(showlegend=True,
                                    height=400,
                                    width=400,
                                    margin={'l':20, 'r': 60, 't': 0, 'b': 0})
    figs_s.update_traces(textposition='outside', textinfo='label+percent')
    st.plotly_chart(figs_s, use_container_width=False)


with colgenv:
    st.subheader('Gender Voice')
    layoutvg = go.Layout(
                        legend=dict(
                        yanchor="top",
            y=1.2,
            xanchor="left",
            x=-0.1)
                        )
    figv_g = go.Figure(data=[go.Pie(labels=v_g.index,
                                                values=v_g['Tweeten'],
                                                hole=.3)], layout=layoutvg)
    figv_g.update_layout(showlegend=True,
                                    height=400,
                                    width=400,
                                    margin={'l':20, 'r': 60, 't': 0, 'b': 0})
    figv_g.update_traces(textposition='outside', textinfo='label+percent')
    st.plotly_chart(figv_g, use_container_width=False)

grouped=tweets_df1.groupby(by=['Gender','Sentiment', 'Compoundscore','Username']).count()[["Text"]].rename(columns={"Text":"Count"})
grouped["Text"] = "Tweets"
grouped = grouped.reset_index()
grouped.head()
# st.dataframe(grouped)
with colsengen: 
    st.subheader('Sentiment Voice Gender Based ')              
    figsemvoice = px.sunburst(grouped,
        path=["Text", "Gender","Sentiment"],
        values='Count',
        width=500, height=500)
    figsemvoice.update_traces(textinfo="label+percent parent")

    st.plotly_chart(figsemvoice, use_container_width=False)
with st.expander('See Data'):
        st.dataframe(grouped)

#################################################################


################################ PLACE ############################
with st.container():
    st.header('Place')  
    colp1, colp2, colp3=st.columns(3)             
    with colp1:
        st.subheader('Sentiment Place')
        df_senman = tweets_df1[['Username', 'Sentiment']].groupby('Sentiment').count()[["Username"]]
        layoutsent = go.Layout(
                    legend=dict(
                    yanchor="top",
        y=1.2,
        xanchor="left",
        x=-0.2)
                    )
        fig_senman = go.Figure(data=[go.Pie(labels=df_senman.index,
                                            values=df_senman['Username'],
                                            hole=.3)], layout=layoutsent)
        fig_senman.update_layout(showlegend=True,
                                height=400,
                                width=400,
                                margin={'l': 20, 'r': 60, 't': 60, 'b': 0})
        fig_senman.update_traces(textposition='outside', textinfo='label+percent')
        st.plotly_chart(fig_senman, use_container_width=False)

    with colp2:
        st.subheader('Gender Place')
        df_gender = tweets_df1[['Username', 'Gender']].groupby('Gender').count()[["Username"]]
        layoutgend = go.Layout(
                    legend=dict(
                    yanchor="top",
        y=1.2,
        xanchor="left",
        x=-0.1)
                    )
        fig_gender = go.Figure(data=[go.Pie(labels=df_gender.index,
                                            values=df_gender['Username'],
                                            hole=.3)], layout=layoutgend)
        fig_gender.update_layout(showlegend=True,
                                height=400,
                                width=400,
                                margin={'l':60, 'r': 60, 't': 0, 'b': 0})
        fig_gender.update_traces(textposition='outside', textinfo='label+percent')
        st.plotly_chart(fig_gender, use_container_width=False)

    with colp3:
        st.subheader('Device Place')

        df_device = tweets_df1[['Username', 'Device']].groupby('Device').count()[["Username"]]
        layoutdev = go.Layout(
                    legend=dict(
            orientation="h",
                    y=-0.8,
                    yanchor='bottom',
                    # xanchor='right',
                    x=0
                     )
                    )
        fig_device = go.Figure(data=[go.Pie(labels=df_device.index,
                                            values=df_device['Username'],
                                            hole=.3)], layout=layoutdev)
        fig_device.update_layout(showlegend=True,
                                height=500,
                                width=420,
                                margin={'l': 60, 'r': 0, 't': 0, 'b':0})
        fig_device.update_traces(textposition='outside', textinfo='label+percent')
        st.plotly_chart(fig_device, use_container_width=False)

    colgeo, colsenp=st.columns([1,1])
    with colgeo:
        st.subheader("Geo Place")
        df_location = tweets_df1[['Username', 'Location']].groupby('Location').count()[["Username"]]
        layout = go.Layout(
                        legend=dict(
                        orientation="h")
                        )
        fig_location = go.Figure(data=[go.Pie(labels=df_location.index,
                                            values=df_location['Username'],
                                            hole=.3
                                            )])
        fig_location.update_layout(showlegend=True,
                                height=400,
                                width=400,
                                margin={'l': 0, 'r': 0, 't': 0, 'b': 20})
        fig_location.update_traces(textposition='inside', textinfo='label+percent')
        st.plotly_chart(fig_location, use_container_width=True)


    # st.map(tweets_df1['Location'])






#################################################
    with colsenp:
        st.subheader('Semanticized Place')
        semplace=tweets_df1.groupby(by=['Gender','Sentiment','Device','Location']).count()[['Username']].rename(columns={'Username':"Count"})
        semplace['Username'] = 'User'
        semplace = semplace.reset_index()
        semplace.head()
    
        figsemplace = px.sunburst(semplace,
            path=["Username", "Gender","Sentiment", 'Device', 'Location'],
            values='Count',
            width=600, height=600)
        figsemplace.update_traces(textinfo="label+percent parent")

        st.plotly_chart(figsemplace, use_container_width=True)
        # with st.expander('See Data'):
        #     st.dataframe(semplace)



##############################################
 



##################################TOPUSERSENTIMENT######################
