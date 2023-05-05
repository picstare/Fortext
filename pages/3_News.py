import streamlit as st
from newsapi import NewsApiClient
import json
import pandas as pd
import numpy as np
import datetime
import nltk
import os


# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

newsapi = NewsApiClient(api_key='28117c7eb02e41088ba0202989e52b20')
newsdatapath='newsdata'



st.set_page_config(page_title='News Analysis', page_icon=':newspaper:', layout='wide')
st.title('News Analysis')
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
 
# Check if filename exists and is not an empty string
    if filename and filename.strip() and os.path.isfile(f"{newsdatapath}/{filename}"):
        with open(f"{newsdatapath}/{filename}", 'r') as f:
            data = json.load(f)
        articles = data['articles']
        articlesdf = pd.DataFrame(articles)
        
     # Define a function to perform NER on a given text

        def extract_locations(text):
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            chunked = nltk.ne_chunk(pos_tags)
            locations = []
            for subtree in chunked.subtrees():
                if subtree.label() == 'GPE':
                    location = ' '.join([leaf[0] for leaf in subtree.leaves()])
                    locations.append(location)
            return locations
    
        articlesdf['locations'] = articlesdf['content'].apply(extract_locations)

        st.dataframe(articlesdf)
        
    