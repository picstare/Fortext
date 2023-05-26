import streamlit as st
from newsapi import NewsApiClient
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



st.set_page_config(page_title='Forteks | News Analysis', page_icon=':newspaper:', layout='wide')
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

# with st.sidebar:

#         filelist = []
#         for root, dirs, files in os.walk("newsdata"):
#             for file in files:
#                 filename = file
#                 filelist.append(filename)
#         # st.write(filelist)

#         optionfile = st.selectbox("Select file:", filelist, index=0)

#         newsdf = pd.read_json("newsdata/" + optionfile)
#         # articledf1['DateTime'] = pd.to_datetime(articledf1['DateTime'], unit='ms')

#         st.write("Choosen file to analyze:", optionfile)


listTabs = [
    "👨‍💼 News Analysis",
    "📈 Data Mining",
    
]

font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 24px;
}
</style>
"""
st.write(font_css, unsafe_allow_html=True)
whitespace = 40
tab1, tab2= st.tabs([s.center(whitespace,"\u2001") for s in listTabs])

with tab1:
    st.header('News Analysis')











with tab2:
    import subprocess
    import streamlit as st
    import json
    import os
    import re
    from bs4 import BeautifulSoup


    SPIDERS = {
        "Antara News": "scrapeprojects/newsscraper/newsscraper/spiders/antarascraper.py",
        "Kompas.com":  "scrapeprojects/newsscraper/newsscraper/spiders/kompascomscraper.py",
        "Detik News":  "scrapeprojects/newsscraper/newsscraper/spiders/detiknews.py",

        # Add more spiders here if needed
    }

    SCRAPY_EXECUTABLE = "C:/Users/JurnalisIndonesia/anaconda3/envs/scrapy_env/Scripts/scrapy"  # Specify the absolute path to the Scrapy executable

    def run_spider(spider_name, query):
        output_folder = "newscraped"
        output_file = f"{output_folder}/{spider_name}-{query}.json"  # Specify the desired output file path with the spider name and query in the file name
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        
        if os.path.exists(output_file):
            # If the file exists, load the existing data
            with open(output_file, "r") as file:
                existing_data = json.load(file)
        else:
            # If the file does not exist, initialize an empty list for data
            existing_data = []
        
        # Temporary output file path
        tmp_output_file = "newscraped/tmp.json"
        
        # Run the spider
        command = [
            SCRAPY_EXECUTABLE,
            "runspider",
            SPIDERS.get(spider_name),
            "-a",
            f"query={query}",
            "-t",
            "jsonlines",
            "-o",
            tmp_output_file,
        ]
        if command:
            subprocess.run(command)
            st.success(f"Spider '{spider_name}' finished running.")
            
            if os.path.exists(tmp_output_file) and os.path.getsize(tmp_output_file) > 0:
                new_data = []
                
                # Load the newly scraped data from the temporary output file line by line
                with open(tmp_output_file, "r") as file:
                    for line in file:
                        # Process each line as a separate JSON object
                        data = json.loads(line)
                        new_data.append(data)
                
                # Append the newly scraped data to the existing data
                existing_data.extend(new_data)
                
                # Write the combined data to the output file
                with open(output_file, "w") as file:
                    json.dump(existing_data, file)
                
                # Remove the temporary output file
                os.remove(tmp_output_file)
            else:
                st.warning("No new data was scraped.")
        else:
            st.error(f"Spider '{spider_name}' not found.")


    st.title("📰 Search News")

    # Form inputs
    with st.form("Spider Form"):
        query = st.text_input("Search:")
        spider_name = st.selectbox("Select Media Outlets:", list(SPIDERS.keys()))
        submitted = st.form_submit_button("Search")

    # Run button
    if submitted:
        if not query:
            st.error("Please enter a query.")
        else:
            st.info(f"Running spider '{spider_name}'...")
            run_spider(spider_name, query)


    folder_path = "newscraped"

    # Get all the files in the folder
    files = os.listdir(folder_path)

    # Filter out only the JSON files
    json_files = [file for file in files if file.endswith(".json")]

    # Loop over the JSON files
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)

        # Load the JSON data from the file
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except (IOError, json.JSONDecodeError) as e:
            st.error(f"Error loading JSON file: {json_file} - {str(e)}")
            continue

        # Display the data in the Streamlit app
        for article in data:
            query = article.get("query")
            title = article.get("title")
            date = article.get("date")
            image_url = article.get("image_url")
            content = article.get("content")
            author = article.get("author")
            # tags = article.get("tags")
            url = article.get("url")

            # Skip this iteration if any essential field is missing
            if None in [title, content, author]:
                continue

            # Remove HTML tags from the content
            soup = BeautifulSoup(content, "html.parser")
            clean_content = re.sub(r"<.*?>", "", soup.get_text())

            # Display the cleaned data
        
            st.write(title)
            st.write(author)
            st.write(date)
            st.write(image_url)
            st.write(clean_content)
            # st.write("Tags:", tags)
            st.write(url)
            st.write("---")








