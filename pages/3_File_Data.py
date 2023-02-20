import time
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
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
from streamlit_tags import st_tags
import altair as alt
import seaborn as sns
from twitter.tokenisasi import Tokenizer
import functools
from genderpred import GndrPrdct
from math import sin
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import train_test_split
import nltk
from googletrans import Translator, constants
import streamlit_nested_layout
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from agepred import get_age
from collections import OrderedDict
from networkx.algorithms import bipartite
from networkx import NetworkXError
from networkx.algorithms.community import greedy_modularity_communities as nxcom
import sys
import pickle
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from pprint import pprint


# sys.stdin.reconfigure(encoding='utf-8')
# sys.stdout.reconfigure(encoding='utf-8')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

model_path = "sklearn/lda_sklearn.pkl"
vectorizer_path = "sklearn/vectorizer.pkl"

st.set_page_config(
    page_title="Twitter.Picanalytics",
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

# with st.container():

df1 = pd.read_pickle("data/psifin.pkl")
    # st.dataframe(df1)
# df2 = pd.read_pickle("data/psi2.pkl")
# df= [df1, df2]
# dfp=pd.concat(df)
# dfp.to_pickle('data/psifin.pkl', compression='infer', protocol=4)




fd1 = pd.read_pickle('data/pdipin.pkl')
# fd2=pd.read_pickle('data/pdip2.pkl')
# fd= [fd1, fd2]
# fpd=pd.concat(fd)
# fpd.reset_index(inplace=True)
# st.dataframe(fpd)

# fpd.to_pickle('data/pdipin.pkl', compression='infer', protocol=4)

flc=fd1.groupby('Date').size().to_frame("Tweets").reset_index() 
dlc= df1.groupby('Date').size().to_frame("Tweets").reset_index()

choose=st.radio('',
    ('Sole', 'Comparation'))

if choose == 'Sole':
     st.line_chart(data=dlc, x='Date', y='Tweets', width=0, height=0, use_container_width=True)

if choose == 'Comparation':

    chart1 = alt.Chart(flc).mark_line().encode(x='Date', y='Tweets', color=alt.value("#E14D2A"))
    chart2 = alt.Chart(dlc).mark_line().encode(x='Date', y='Tweets', color=alt.value("#DC0000"))
    st.altair_chart(chart1 + chart2, use_container_width=True)

st.title("Graph Network Analysis")

G = nx.MultiDiGraph()

for r in df1.iterrows():
    G.add_node(r[1]['User'], gender=r[1]['Gender'], color=r[1]['Colorgend'], device=r[1]['Device'], location=r[1]["Location"])
    for user in r[1]['Splitmentions']:
        G.add_edge(r[1]['User'], user, label='@')
    for user in r[1]['Splithast']:
        G.add_edge(r[1]['User'], user, weight=2, color='grey', label='#')
    for keyword in r[1]['Keyword']:
        G.add_edge(r[1]['User'], keyword, color='green', label='topic')

# for r in df2.itterows():
#     G.add_node(r[1]['User'], gender=r[1]['Gender'], color=r[1]['Colorgend'], device=r[1]['Device'], location=r[1]["Location"])
#     for user in r[1]['Splitmentions']:
#         G.add_edge(r[1]['User'], user, label='@')
#     for user in r[1]['Splithast']:
#         G.add_edge(r[1]['User'], user, weight=2, color='grey', label='#')
#     for keyword in r[1]['Keyword']:
#         G.add_edge(r[1]['User'], keyword, color='green', label='topic')
     

            
                

if G.has_node(''):
    G.remove_node('')

# list(G.edges())[:]
# list(G.nodes(data=True))[:]
        
            
     
nt=Network('800px', '100%', notebook=True, directed=True, neighborhood_highlight=True, select_menu=True,filter_menu=True, cdn_resources='remote')
nt.show_buttons(filter_=['physics'])
        
        #    # print(neighbor_map)
        

nt.from_nx(G)
neighbor_map= nt.get_adj_list()
for node in nt.nodes:
        node['value']=len(neighbor_map[node['id']])
        node['title']=' Neighbors:<br>' + '<br>'.join(neighbor_map[node['id']])
        # node['color']= color_map[node['id']]

            # degcent = nx.degree_centrality(G)
            # # eigen_vector=nx.eigenvector_centrality(G)
            # betwen= nx.betweenness_centrality(G)
            # closeness=nx.closeness_centrality(G)
            # indeg=nx.in_degree_centrality(G)

            # nx.set_node_attributes(G, degcent, "centrality")

            # c = nxcom(G)
            # # Count the communities
            # len(c)

            # try:
            #     # Find and print node sets
            #     left, right = bipartite.sets(G)
            #     print("Left nodes\n", left)
            #     print("\nRight nodes\n", right)
            # except NetworkXError as e:
            #     # Not an affiliation network
            #     st.write(e)


            
            
           # try:
           # path = '/tmp'
           # nt.save_graph(f'{path}/pyvis_graph.html')
           # HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

           # except:
path ='html_files'

nt.save_graph(f'{path}/pyvis_graph.html')
HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding="utf-8")


components.html(HtmlFile.read(), height=1500, scrolling=False)