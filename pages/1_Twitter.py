import streamlit as st
import pandas as pd
import numpy as np
from utils import logout
from streamlit_extras.switch_page_button import switch_page
import os
import json
import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyvis.network import Network as net
import tweepy
from streamlit_tags import st_tags
from json import JSONEncoder
import PIL
import time
import nltk
import gensim
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pyLDAvis.gensim
from matplotlib import font_manager, mathtext
from pathlib import Path

nltk.download('vader_lexicon')


consumer_key = "1Hov50UKDBETZmY1wR9zkE3Q7"
consumer_secret = "lAAcJVSDE1Oyc1BuOmxJNN4D575NkHQcg3hEa5zeurrGwCpXH0"
access_token = "16645853-jRxQql8XCzcaWsBSTeA3eutXPA5xRcHxqRHDgx6m9"
access_token_secret = "STmNDeF9BX33PTYuE18vPq7yndA4okKroeq9LXX6FV2gk"
bearer_token = 'AAAAAAAAAAAAAAAAAAAAALMe9wAAAAAAN%2BggvuMVDLKLIEX3Kk%2B8nOSxH88%3DiBMoMLAjE4JuPUKzRyZjYbs5zRZ82uZk9T89YCBgKkeXmgbKY5'

accounts = []
auth = tweepy.OAuth1UserHandler(consumer_key=consumer_key, consumer_secret=consumer_secret, access_token=access_token, access_token_secret=access_token_secret)
# auth.set_access_token(, 'ph47iEpbfD4USmwynPCL1LLNtl7f9seLovjIHOUqwTuQq')
api = tweepy.API(auth)
client = tweepy.Client(bearer_token=bearer_token)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


st.set_page_config(page_title="Forteks | Twitter Analysis", layout="wide")

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div.vis-configuration-wrapper{display: block; width: 400px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

class DateTimeEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)



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
    st.title("Twitter Analysis")
###############################################
listTabs = [
    "👨‍💼 Key Persons Analysis",
    "🦈 Issue Analysis",
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
whitespace = 30
tab1, tab2, tab3 = st.tabs([s.center(whitespace,"\u2001") for s in listTabs])

with tab1:
    st.header("KEY PERSONS")

    container1=st.container()
    with container1:
        folder_path = "twittl"
        files = os.listdir(folder_path)

        # Get the modification times of the files
        file_times = [(f, os.path.getmtime(os.path.join(folder_path, f))) for f in files]

        # Sort the files based on modification time in descending order
        sorted_files = sorted(file_times, key=lambda x: x[1], reverse=True)

        # Select the four newest files
        num_files = min(4, len(sorted_files))
        newest_files = [f[0] for f in sorted_files[:num_files]]

        # Update the 'files' variable with the names of the newest files
        files = newest_files

        if len(files) > 0:
            # Create a Streamlit column for each file
            cols = st.columns(num_files)

            for i, col in enumerate(cols):
                # Check if the file is in JSON format
                if i < num_files and files[i].endswith('.json'):
                    # Open the file and read its contents as a JSON object
                    with open(os.path.join(folder_path, files[i]), 'r') as f:
                        user_data = json.load(f)
                        
                    # Access the follower count
                    followers_count = user_data["tweets"][0]["user"]["followers_count"]
                    profilepic=user_data["tweets"][0]["user"]["profile_image_url_https"]

                    friend_count=user_data["tweets"][0]["user"]["friends_count"]
                    listed_count=user_data["tweets"][0]["user"]["listed_count"]
                    status=user_data["tweets"][0]["user"]["statuses_count"]

                    
                    # Display the user data in the column
                    col.image(profilepic, width=100)
                    # col.write(f"Account: {files[i].replace('_data.json', '')}")
                    col.write(f"{user_data['name']}")
                    col.write(f"{user_data['description']}")
                    col.write(f"Tweets: {status}")
                    col.write(f"Followers: {followers_count}")
                    col.write(f"Friend: {friend_count}")
                    col.write(f"Listed: {listed_count}")

        ######################################CHART TIME SERIES#######################

        st.header("TIME SERIES ANALYSIS OF THE KEY PERSONs")
        

        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
        # data = []
        df1 = None

        if files:
            data = []
            for file in files:
                with open(file, 'r') as f:
                    file_data = json.load(f)
                    for tweet in file_data['tweets']:
                        data.append({
                            'name': tweet['user']['name'],
                            'date': pd.to_datetime(tweet['created_at'])
                        })
            df1 = pd.DataFrame(data)

        # Create a list of available screen names
        if df1 is not None:
            names = list(df1['name'].unique())
        else:
            names = []

        # Set the default selected names to the first 4 accounts
        default_names = names[:4]

        # Set the default time range to one month from the current date
        end_date = pd.to_datetime(datetime.today(), utc=True)
        start_date = end_date - timedelta(days=30)

        # Create widgets for selecting the screen name and time range
        selected_names = st.multiselect('Select names to compare', names, default=default_names, key='selper')
        cols_ta, cols_tb = st.columns([1, 1])
        start_date = pd.to_datetime(cols_ta.date_input('Start date', value=start_date), utc=True)
        end_date = pd.to_datetime(cols_tb.date_input('End date', value=end_date), utc=True)

        # Filter the data based on the selected names and time range
        if df1 is not None:
            mask = (df1['name'].isin(selected_names)) & (df1['date'] >= start_date) & (df1['date'] <= end_date)
            df1_filtered = df1.loc[mask]
        else:
            df1_filtered = pd.DataFrame()

        if len(df1_filtered) > 0:
            df1_grouped = df1_filtered.groupby(['date', 'name']).size().reset_index(name='count')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df1_grouped, x='date', y='count', hue='name', ax=ax)
            ax.set_title(f"Tweets per Day for {', '.join(selected_names)}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Tweets")
            st.pyplot(fig)
        else:
            st.write("No data available for the selected time range and users.")
    
        st.markdown("---")

    #####################SNA########################
        st.header("SOCIAL NETWORK ANALYSIS OF THE KEY PERSONS")
        # folder_path = 'twittl/'
        def get_followers_following_tweets(folder_path):
            followers = {}
            following = {}
            tweet_data = []

            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    account = file_name.split('.')[0]
                    with open(os.path.join(folder_path, file_name), 'r') as f:
                        data = json.load(f)
                        followers[account] = data['followers']
                        following[account] = data['following']
                        tweets = data['tweets']
                        for tweet in tweets:
                            tweet_info = {}
                            tweet_info['id_str'] = tweet['id_str']
                            tweet_info['created_at'] = tweet['created_at']
                            tweet_info['full_text'] = tweet['full_text']
                            tweet_info['user_mentions'] = tweet['entities']['user_mentions']
                            tweet_info['retweeted_user'] = tweet['retweeted_status']['user']['screen_name'] if 'retweeted_status' in tweet else None
                            tweet_info['in_reply_to_screen_name'] = tweet['in_reply_to_screen_name']
                            tweet_info['tweet_url'] = f"https://twitter.com/{account}/status/{tweet['id_str']}"
                            tweet_data.append(tweet_info)

            return followers, following, tweet_data

        def build_social_network(followers, following):
            G = nx.DiGraph()

            for account in followers.keys():
                G.add_node(account, title=account, label=account)

                for follower in followers[account]:
                    G.add_edge(follower, account)

                for followee in following[account]:
                    G.add_edge(account, followee)

                # Add 'not_followed_back' nodes and edges
                not_followed_back = set(followers[account]) - set(following[account])
                for not_followed in not_followed_back:
                    G.add_node(not_followed, title=not_followed, label=not_followed)
                    G.add_edge(not_followed, account, relationship='not_followed_back')

                # Add 'not_following_back' nodes and edges
                not_following_back = set(following[account]) - set(followers[account])
                for not_following in not_following_back:
                    G.add_node(not_following, title=not_following, label=not_following)
                    G.add_edge(account, not_following, relationship='not_following_back')

            return G
        

        def visualize_social_network(G, selected_accounts):
            subgraph_nodes = set()
            for account in selected_accounts:
                subgraph_nodes |= set([account] + followers[account] + following[account])
            subgraph = G.subgraph(subgraph_nodes)

            nt = net(height='750px', width='100%', bgcolor='#fff', font_color='#3C486B', directed=True)

            node_colors = {}
            for account in selected_accounts:
                node_colors[account] = '#2CD3E1'
                for follower in followers[account]:
                    node_colors[follower] = '#FF6969'
                for followee in following[account]:
                    node_colors[followee] = '#FFD3B0'

                # Add node colors for 'not_following_back'
                not_following_back = set(following[account]) - set(followers[account])
                for not_following in not_following_back:
                    node_colors[not_following] = '#F5AEC1'

                # Add node colors for 'not_followed_back'
                not_followed_back = set(followers[account]) - set(following[account])
                for not_followed in not_followed_back:
                    node_colors[not_followed] = '#FFA500'

            for node in subgraph.nodes():
                nt.add_node(node, title=node, label=node, color=node_colors.get(node, 'skyblue'))

            for edge in subgraph.edges():
                nt.add_edge(edge[0], edge[1])

            nt.font_color = 'white'
        
            nt.save_graph('html_files/social_network.html')

            # Display the network visualization in Streamlit
            with open('html_files/social_network.html', 'r') as f:
                html_string = f.read()
                st.components.v1.html(html_string, height=960, scrolling=True)

        # Read the data
        followers, following, tweet_data = get_followers_following_tweets(folder_path)

        # Build the social network
        G = build_social_network(followers, following)

        default_accounts = list(followers.keys())[:4]

            # Ask the user which accounts to visualize using st.sidebar.multiselect
        selected_accounts = st.multiselect('Select accounts to visualize', list(followers.keys()), default=default_accounts)

        # Retrieve the account names instead of file names
        account_names = [account.split('_')[0] for account in selected_accounts]

        # Display the selected account names in the Streamlit header
        st.header("Social Network Accounts' Followers and Friends: " + ', '.join(account_names))
        # Visualize the selected accounts
        visualize_social_network(G, selected_accounts)   
    
    st.markdown("---")
 ##############################################################################
 ############################################################################                   
                    
with tab2:
    st.header('Issue Analysis')

    folder_path = "twitkeys"

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    df = None

    if files:
        data = []
        for file in files:
            keyword = os.path.splitext(os.path.basename(file))[0]
            with open(file, 'r') as f:
                file_data = json.load(f)
                for tweet_data in file_data['data']:
                    tweet = tweet_data['Text']
                    created_at = pd.to_datetime(tweet_data['Created At'])
                    data.append({
                        'keyword': keyword,
                        'text': tweet,
                        'date': created_at
                    })
        df = pd.DataFrame(data)

    # Create a list of available keywords
    if df is not None:
        keywords = list(df['keyword'].unique())
    else:
        keywords = []

    # Set the default selected keywords to the first 4 keywords
    default_keywords = keywords[:4]

    # Set the default time range to one month from the current date
    end_date = pd.to_datetime(datetime.today().date())
    start_date = end_date - timedelta(days=30)

    # Create widgets for selecting the keywords and time range
    selected_keywords = st.multiselect('Select keywords to compare', keywords, default=default_keywords, key='selissue')
    cols_ta, cols_tb = st.columns([1, 1])
    start_date = pd.to_datetime(cols_ta.date_input('Start date', value=start_date, key='start_date')).date()
    end_date = pd.to_datetime(cols_tb.date_input('End date', value=end_date, key='end_date')).date()

    # Filter the data based on the selected keywords and time range
    if df is not None:
        mask = (df['keyword'].isin(selected_keywords)) & (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
        df_filtered = df.loc[mask]
    else:
        df_filtered = pd.DataFrame()

    if len(df_filtered) > 0:
        df_grouped = df_filtered.groupby(['date', 'keyword']).size().reset_index(name='count')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df_grouped, x='date', y='count', hue='keyword', ax=ax)
        ax.set_title(f"Tweets per Day for {', '.join(selected_keywords)}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Tweets")
        st.pyplot(fig)
    else:
        st.write("No data available for the selected time range and keywords.")

    
    ################ SNA ####################
    import glob


    def process_json_files(files):
        G = nx.DiGraph()
    
        for file in files:
            with open(file, 'r') as f:
                file_data = json.load(f)
                for tweet_data in file_data['data']:
                    user_screen_name = tweet_data['User Screen Name']
                    mentioned_users = [user['User Screen Name'] for user in tweet_data['mentioned_users']]
                    retweeted_user = tweet_data['Retweeted Tweet']['Author Screen Name'] if 'Retweeted Tweet' in tweet_data else None
                    hashtags = tweet_data['Hashtags']
                    
                    G.add_node(user_screen_name)
                    if retweeted_user:
                        G.add_node(retweeted_user)
                        G.add_edge(user_screen_name, retweeted_user, relationship='retweeted')
                    for mentioned_user in mentioned_users:
                        G.add_node(mentioned_user)
                        G.add_edge(user_screen_name, mentioned_user, relationship='mentioned')

                    for hashtag in hashtags:
                    # Connect users who have mentioned or used the same hashtag
                        users_with_same_hashtag = [node for node in G.nodes if G.nodes[node].get('relationship') == 'mentioned' and hashtag in G.edges[user_screen_name, node]['relationship']]
                        for user in users_with_same_hashtag:
                            G.add_edge(user_screen_name, user, relationship=hashtag)
        
        return G

    # Read JSON files from the folder
    folder_path = "twitkeys"
    file_path = os.path.join("twitkeys", f"{keyword}.json")
    # files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    # files = [os.path.splitext(os.path.basename(file))[0] for file in glob.glob(os.path.join(folder_path, '*.json'))]

    # Create the social network graph
    G = process_json_files(files)

    # Function to visualize the social network using pyvis
    def visualize_social_network(G):
        nt = net(height='750px', width='100%', bgcolor='#ffffff', font_color='#333333', directed=True)

        for node in G.nodes:
            nt.add_node(node, label=node)

        for edge in G.edges:
            source = edge[0]
            target = edge[1]
            relationship = G.edges[edge]['relationship']
            nt.add_edge(source, target, label=relationship)

        nt.save_graph('html_files/issue_social_network.html')

    # Display the graph in Streamlit
        with open('html_files/issue_social_network.html', 'r') as f:
            html_string = f.read()
            st.components.v1.html(html_string, height=960, scrolling=True)

    default_files = [os.path.splitext(os.path.basename(file))[0] for file in files[:4]] if len(files) >= 4 else [os.path.splitext(os.path.basename(file))[0] for file in files]
    selected_files = st.multiselect('Select Issue/Topic', [os.path.splitext(os.path.basename(file))[0] for file in files], default=default_files, format_func=lambda x: f"{x}.json")

    # Process the selected JSON files and build the social network graph
    selected_files_paths = [os.path.join(folder_path, f"{file}.json") for file in selected_files]
    selected_G = process_json_files(selected_files_paths)

    # Visualize the social network
    # visualize_social_network(selected_G)

    ####################DEGREE CENTRALITY###########################
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(selected_G)

    # Create a subgraph with nodes having non-zero degree centrality
    degree_subgraph = selected_G.subgraph([node for node, centrality in degree_centrality.items() if centrality > 0])

    # Function to visualize the degree centrality network and top actors using Pyvis and matplotlib
    def visualize_degree_centrality_network(subgraph, centrality_values):
        nt = net(height='750px', width='100%', bgcolor='#ffffff', font_color='#333333', directed=True)

        # Add nodes to the network with size based on degree centrality
        for node in subgraph.nodes:
            centrality = centrality_values[node]
            node_size = centrality * 20  # Adjust the scaling factor as needed
            nt.add_node(node, label=node, size=node_size)

        # Add edges to the network
        for edge in subgraph.edges:
            source = edge[0]
            target = edge[1]
            relationship = subgraph.edges[edge]['relationship']
            nt.add_edge(source, target, label=relationship)

        nt.save_graph('html_files/degree_centrality_network.html')

        

        with open('html_files/degree_centrality_network.html', 'r') as f:
            html_string = f.read()
            st.components.v1.html(html_string, height=960, scrolling=True)

        # Calculate and plot the top actors based on degree centrality
        top_actors = sorted(centrality_values, key=centrality_values.get, reverse=True)[:5]
        centrality_scores = [centrality_values[actor] for actor in top_actors]

        y_pos = np.arange(len(top_actors))

        fig, ax = plt.subplots()
        ax.barh(y_pos, centrality_scores)
        ax.set_xlabel('Degree Centrality')
        ax.set_ylabel('Top Actors')
        ax.set_title('Top Main Actors')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_actors)
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)


################## BETWEENESS CENTRALITY ##########################
    # Calculate betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(selected_G)

    # Create a subgraph with nodes having non-zero betweenness centrality
    betweenness_subgraph = selected_G.subgraph([node for node, centrality in betweenness_centrality.items() if centrality > 0])



    # Function to visualize the betweenness centrality network and top actors using Pyvis and matplotlib
    def visualize_betweenness_centrality_network(subgraph, centrality_values):
        nt = net(height='750px', width='100%', bgcolor='#ffffff', font_color='#333333', directed=True)

        # Add nodes to the network with size based on betweenness centrality
        for node in subgraph.nodes:
            centrality = centrality_values[node]
            node_size = centrality * 20  # Adjust the scaling factor as needed
            nt.add_node(node, label=node, size=node_size)

        # Add edges to the network
        for edge in subgraph.edges:
            source = edge[0]
            target = edge[1]
            relationship = subgraph.edges[edge]['relationship']
            nt.add_edge(source, target, label=relationship)

        nt.save_graph('html_files/betweenness_centrality_network.html')

        
        with open('html_files/betweenness_centrality_network.html', 'r') as f:
            html_string = f.read()
            st.components.v1.html(html_string, height=960, scrolling=True)

        # Calculate and plot the top actors based on betweenness centrality
        top_actors = sorted(centrality_values, key=centrality_values.get, reverse=True)[:5]
        centrality_scores = [centrality_values[actor] for actor in top_actors]

        y_pos = np.arange(len(top_actors))

        fig, ax = plt.subplots()
        ax.barh(y_pos, centrality_scores)
        ax.set_xlabel('Betweenness Centrality')
        ax.set_ylabel('Actors')
        ax.set_title('Top Actors based on Betweenness Centrality')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_actors)
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)
##################################################################
    def calculate_closeness_centrality(G):
        closeness_centrality = nx.closeness_centrality(G)

        return closeness_centrality
    
      # Calculate closeness centrality
    closeness_centrality = calculate_closeness_centrality(selected_G)

        # Sort the centrality values in descending order
    sorted_centrality = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)

        # Extract the top nodes and their centrality values
    top_nodes = [node for node, centrality in sorted_centrality[:10]]
    top_centrality = [centrality for node, centrality in sorted_centrality[:10]]
        # Function to visualize the closeness centrality network using Pyvis
    def visualize_closeness_centrality_network(G, centrality_values):
        nt = net(height='750px', width='100%', bgcolor='#ffffff', font_color='#333333', directed=True)

        # Add nodes to the network with size based on closeness centrality
        for node in G.nodes:
            centrality = centrality_values[node]
            node_size = centrality * 20  # Adjust the scaling factor as needed
            nt.add_node(node, label=node, size=node_size)

        # Add edges to the network
        for edge in G.edges:
            source = edge[0]
            target = edge[1]
            relationship = G.edges[edge]['relationship']
            nt.add_edge(source, target, label=relationship)

        nt.save_graph('html_files/closeness_centrality_network.html')
        with open('html_files/closeness_centrality_network.html', 'r') as f:
            html_string = f.read()
            st.components.v1.html(html_string, height=960, scrolling=True)

        plt.figure(figsize=(10, 6))
        plt.bar(top_nodes, top_centrality)
        plt.xlabel('Nodes')
        plt.ylabel('Closeness Centrality')
        plt.title('Top Actors Based on Closeness Centrality')
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Display the chart in Streamlit
        st.pyplot(plt)




############################VISUGRAPJH###########################
    colviz1, colviz2, colviz3, colviz4=st.tabs(['Social Network','Aktor Utama', 'Aktor Penjembatan', 'Aktor Pendukung'])
    with colviz1:
         visualize_social_network(selected_G)
        
    with colviz2:
        visualize_degree_centrality_network(degree_subgraph, degree_centrality)

    with colviz3:
        visualize_betweenness_centrality_network(betweenness_subgraph, betweenness_centrality)
    
    with colviz4:
        visualize_closeness_centrality_network(selected_G, closeness_centrality)
    
#################################################################################
######################TOPIC MODELING#############################################



    # Function to load and preprocess the text data
    def load_and_preprocess_data(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Preprocess the text (e.g., lowercase, tokenization, stopwords removal, etc.)
        preprocessed_text_data = []
        stop_words = set(stopwords.words('indonesian'))
        lemmatizer = WordNetLemmatizer()

        for tweet_data in data['data']:
            text = tweet_data['Text']

            # Convert text to lowercase
            text = text.lower()

            # Tokenize the text
            tokens = word_tokenize(text)

            # Remove stopwords and perform lemmatization
            processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

            # Append the processed tokens to the preprocessed text data
            preprocessed_text_data.append(processed_tokens)
        
        # Return the preprocessed text data
        return preprocessed_text_data

    # Function to perform topic modeling
    def perform_topic_modeling(text_data):
        # Create a dictionary from the text data
        dictionary = gensim.corpora.Dictionary(text_data)
        
        # Create a corpus (Bag of Words representation)
        corpus = [dictionary.doc2bow(text) for text in text_data]
        
        # Perform topic modeling using LDA
        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)
        
        # Return the LDA model
        return lda_model

    

    # Main Streamlit app
    st.title("Topic Modeling with pyLDAvis")

    # Folder path containing the JSON files
    folder_path = "twitkeys"

    # Get the list of JSON files in the folder
    file_list = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.json')]

    # Select files
    selected_files = st.multiselect("Select Files", file_list, default=file_list[:1])


    # List to store preprocessed text data from selected files
    preprocessed_text_data = []

    # Iterate over the selected files
    for file_name in selected_files:
        file_path = os.path.join(folder_path, file_name)

        # Load and preprocess the text data from the file
        text_data = load_and_preprocess_data(file_path)

        # Append the preprocessed text data to the list
        preprocessed_text_data.extend(text_data)


        # Perform topic modeling on the preprocessed text data
        lda_model = perform_topic_modeling(preprocessed_text_data)

        dictionary = gensim.corpora.Dictionary(preprocessed_text_data)
        corpus = [dictionary.doc2bow(text) for text in preprocessed_text_data]

        # Generate the pyLDAvis visualization
        lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)

        # Save the HTML file
        pyLDAvis.save_html(lda_display, "lda.html")

        # Read the HTML file
        with open("lda.html", "r") as f:
            html_string = f.read()

        # Display the HTML file in Streamlit
        st.components.v1.html(html_string, height=800, width=1500, scrolling=False)


################################ SENTIMENT ANALYSIS#################################



    from nltk.sentiment import SentimentIntensityAnalyzer
    import numpy as np

    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


    # Function to load and preprocess the text data
    def load_and_preprocess_data(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Preprocess the text (e.g., lowercase, tokenization, stopwords removal, etc.)
        preprocessed_text_data = []
        stop_words = set(stopwords.words('indonesian'))
        stemmer = StemmerFactory().create_stemmer()

        for tweet_data in data['data']:
            text = tweet_data['Text']

            # Convert text to lowercase
            text = text.lower()

            # Tokenize the text
            tokens = word_tokenize(text)

            # Remove stopwords and perform stemming
            processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

            # Append the processed tokens to the preprocessed text data
            preprocessed_text_data.append(processed_tokens)

        # Return the preprocessed text data
        return preprocessed_text_data

    # Function to perform sentiment analysis using VADER
    def perform_sentiment_analysis(text_data):
        # Initialize the VADER sentiment intensity analyzer
        sia = SentimentIntensityAnalyzer()

        # Perform sentiment analysis on each text
        sentiment_scores = []
        for text in text_data:
            sentiment_score = sia.polarity_scores(' '.join(text))
            sentiment_scores.append(sentiment_score)

        # Convert the sentiment scores to a DataFrame
        df_sentiment = pd.DataFrame(sentiment_scores)

        # Return the DataFrame with sentiment scores
        return df_sentiment

    # Main Streamlit app
    st.title("Sentiment Analysis with VADER for Bahasa Indonesia")

    # Folder path containing the JSON files
    folder_path = "twitkeys"

    # Get the list of JSON files in the folder
    file_list = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.json')]

    # Select files
    selected_files = st.multiselect("Select Files", file_list, default=file_list[:1], key="file_selector")

    # Iterate over the selected files
    for file_name in selected_files:
        # List to store preprocessed text data from the current file
        preprocessed_text_data = []

        # File path of the current file
        file_path = os.path.join(folder_path, file_name)

        # Load and preprocess the text data from the file
        text_data = load_and_preprocess_data(file_path)

        # Append the preprocessed text data to the list
        preprocessed_text_data.extend(text_data)

        # Perform sentiment analysis on the preprocessed text data
        df_sentiment = perform_sentiment_analysis(preprocessed_text_data)

        # Display the sentiment analysis results
        st.subheader(f"Sentiment Analysis: {file_name}")
        st.dataframe(df_sentiment)

        # Calculate the sentiment distribution for the current file
        sentiment_distribution = df_sentiment.mean().drop("compound")

        # Plot the sentiment distribution as a pie chart
        plt.figure()
        plt.pie(sentiment_distribution.values, labels=sentiment_distribution.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f"Sentiment Distribution: {file_name}")
        plt.tight_layout()
        st.pyplot(plt)


################################# SENTIMENT ANALYSIS PER USER PER FILES  ##############################
    def load_and_preprocess_data(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Preprocess the text (e.g., lowercase, tokenization, stopwords removal, etc.)
        preprocessed_text_data = []
        stop_words = set(stopwords.words('indonesian'))
        stemmer = StemmerFactory().create_stemmer()

        for tweet_data in data['data']:
            text = tweet_data['Text']
            user = tweet_data['User Name']  # Get the user name

            # Convert text to lowercase
            text = text.lower()

            # Tokenize the text
            tokens = word_tokenize(text)

            # Remove stopwords and perform stemming
            processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

            # Append the processed tokens and the user to the preprocessed text data
            preprocessed_text_data.append((processed_tokens, user))

        # Return the preprocessed text data
        return preprocessed_text_data

       # Perform sentiment analysis per user
    def perform_sentiment_analysis_per_user(text_data):
        # Initialize the VADER sentiment intensity analyzer
        sia = SentimentIntensityAnalyzer()

        # Create a dictionary to store sentiment scores per user
        user_sentiment_scores = {}

        # Perform sentiment analysis on each text per user
        for text, user in text_data:
            sentiment_score = sia.polarity_scores(' '.join(text))

            # Add the sentiment score to the user's scores
            if user not in user_sentiment_scores:
                user_sentiment_scores[user] = {
                    'positive': [],
                    'negative': [],
                    'neutral': [],
                    'compound': []
                }

            user_sentiment_scores[user]['positive'].append(sentiment_score['pos'])
            user_sentiment_scores[user]['negative'].append(sentiment_score['neg'])
            user_sentiment_scores[user]['neutral'].append(sentiment_score['neu'])
            user_sentiment_scores[user]['compound'].append(sentiment_score['compound'])

        # Calculate the average sentiment scores per user
        user_sentiment_scores_avg = {}
        for user, scores in user_sentiment_scores.items():
            user_sentiment_scores_avg[user] = {
                'positive': np.mean(scores['positive']),
                'negative': np.mean(scores['negative']),
                'neutral': np.mean(scores['neutral']),
                'compound': np.mean(scores['compound'])
            }

        # Convert the sentiment scores to a DataFrame
        df_sentiment_per_user = pd.DataFrame.from_dict(user_sentiment_scores_avg, orient='index')

        # Return the DataFrame with sentiment scores per user
        return df_sentiment_per_user

    # Main Streamlit app
    st.title("Sentiment Analysis per user with VADER for Bahasa Indonesia")

    # Folder path containing the JSON files
    folder_path = "twitkeys"

    # Get the list of JSON files in the folder
    file_list = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.json')]

    # Select files
    selected_files = st.multiselect("Select Files", file_list, default=file_list[:1], key="file_selector_sent")

    # Iterate over the selected files
    for file_name in selected_files:
        # List to store preprocessed text data from the current file
        preprocessed_text_data = []

        # File path of the current file
        file_path = os.path.join(folder_path, file_name)

        # Load and preprocess the text data from the file
        text_data = load_and_preprocess_data(file_path)

        # Append the preprocessed text data to the list
        preprocessed_text_data.extend(text_data)

        # Perform sentiment analysis per user on the preprocessed text data
        df_sentiment_per_user = perform_sentiment_analysis_per_user(preprocessed_text_data)

        # Display the sentiment analysis results per user
        st.subheader(f"Sentiment Analysis per User: {file_name}")
        st.dataframe(df_sentiment_per_user)

        # Plot the sentiment scores per user as a bar chart
        ax = df_sentiment_per_user.plot(kind='bar', rot=0)
        plt.xlabel('User')
        plt.ylabel('Sentiment Score')
        plt.title(f"Sentiment Analysis per User: {file_name}")
        plt.xticks(rotation='vertical')
        plt.tight_layout()

        # Modify the text of user in the bar chart
        for p in ax.patches:
            ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=5)

        st.pyplot(plt)
    

######################### LOCATION ##############################

   
    import folium
    from streamlit_folium import folium_static, st_folium
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderUnavailable
    
    
    # Create a geolocator object
    geolocator = Nominatim(user_agent='twitter_map_app')

    # Define a function to perform geocoding with caching
    @st.cache_data
    def geocode_location(location):
        try:
            location_data = geolocator.geocode(location, timeout=5)  # Increase the timeout value as needed
            if location_data:
                return location_data.latitude, location_data.longitude
        except GeocoderUnavailable:
            st.warning(f"Geocoding service is unavailable for location: {location}")
        return None, None

    # Get the file paths of all JSON files in the "twitkeys" folder
    file_paths = glob.glob('twitkeys/*.json')

    # Sort the file paths by modification time (newest to oldest)
    file_paths.sort(key=os.path.getmtime, reverse=True)

    # Select the four newest files
    default_files = file_paths[:1]

    # Allow users to select multiple files using a multiselect widget
    selected_files = st.multiselect("Select JSON Files", file_paths, default=default_files)

    # Define variables to store the min/max latitude and longitude
    min_latitude = float('inf')
    max_latitude = float('-inf')
    min_longitude = float('inf')
    max_longitude = float('-inf')

    # Iterate over the selected files
    for file_path in selected_files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        user_data = data['data']

        # Perform geocoding for each user location
        for user in user_data:
            location = user.get('User Location')
            if location:
                latitude, longitude = geocode_location(location)
                user['Latitude'] = latitude
                user['Longitude'] = longitude
                time.sleep(1)  # Add a 1-second delay between requests

                # Update the min/max latitude and longitude
                if latitude is not None:
                    min_latitude = min(min_latitude, latitude)
                    max_latitude = max(max_latitude, latitude)
                if longitude is not None:
                    min_longitude = min(min_longitude, longitude)
                    max_longitude = max(max_longitude, longitude)

    # Calculate the center latitude and longitude
    center_latitude = (min_latitude + max_latitude) / 2
    center_longitude = (min_longitude + max_longitude) / 2

    # Create a Folium map object
    m = folium.Map(location=[center_latitude, center_longitude], zoom_start=2)

    # Add markers to the map
    for user in user_data:
        latitude = user.get('Latitude')
        longitude = user.get('Longitude')
        user_name = user.get('User Name')

        if latitude is not None and longitude is not None:
            popup = f"User: {user_name}\nLocation: {user['User Location']}"
            folium.Marker([latitude, longitude], popup=popup, tooltip=user_name).add_to(m)

    # Display the map for the current file
    st.header(f"User Map: {file_path}")
    st_folium(m, width=1500, height=600)

#################################################################
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    import joblib
    from sklearn.metrics import accuracy_score
    import plotly.express as px
    
    import numpy as np

    st.title("Gender Prediction from Twitter Data")
    st.header("Predicted Gender")
    
    def preprocess_tweet(tweet, training_columns):
        processed_tweet = {
            'User Name': str(tweet[0]),
            'User Description': str(tweet[1]).lower() if tweet[1] else '',
            'Text': str(tweet[2]).lower(),
        }

        processed_tweet = {k: float(v) if isinstance(v, str) and v.isnumeric() else v for k, v in processed_tweet.items()}
        processed_tweet = pd.DataFrame(processed_tweet, index=[0])

        # Perform one-hot encoding on the categorical variables
        processed_tweet_encoded = pd.get_dummies(processed_tweet)
        processed_tweet_encoded = processed_tweet_encoded.reindex(columns=training_columns, fill_value=0)

        return processed_tweet_encoded.values.flatten()

    
    def predict_gender(model, features, training_columns):
        processed_tweet = preprocess_tweet(features, training_columns)
        prediction = model.predict([processed_tweet])
        return prediction[0]



    dfout = pd.read_json('output1.json')

    # st.dataframe(dfout)
    # print ("DFOUT:",dfout)

    # Prepare the data
    if 'Gender' in dfout.columns:
        X = dfout.drop('Gender', axis=1)
    else:
        X = dfout.copy()

    if 'Gender' in dfout.columns:
        y = dfout['Gender']
    else:
        # Handle the case when 'Gender' column is missing
        # For example, you can print an error message or take appropriate action
        st.write('Gender not in df.columns')

    # Perform one-hot encoding on the categorical variables in X
    X_encoded = pd.get_dummies(X)

    # Get the training columns from the X_encoded DataFrame
    training_columns = X_encoded.columns

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Create a Gradient Boosting Classifier model
    model = GradientBoostingClassifier()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict the gender for the test data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    st.write('Accuracy:', accuracy)

    # Save the trained model to a file
    joblib.dump(model, 'modelgend.pkl')

    # Get the file paths of all JSON files in the "twitkeys" folder
    file_paths = glob.glob('twitkeys/*.json')

    # Sort the file paths by modification time (newest to oldest)
    file_paths.sort(key=os.path.getmtime, reverse=True)

    # Allow users to select multiple files using a multiselect widget
    
    selected_files = st.multiselect("Select JSON Files", file_paths, default=file_paths[:4], key='gensel')

    data_list = []

    # Define the number of columns based on the number of selected files
    num_columns = len(selected_files)

    # Create a grid layout with the specified number of columns
    columns = st.columns(num_columns)

    for i, file_path in enumerate(selected_files):
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            data_list.extend(json_data["data"])

        dfgend = pd.DataFrame(data_list)
        # Drop irrelevant columns
        columns_to_drop = ["User Screen Name", "User Location", "Hashtags", "Source", "in_reply_to_name", "mentioned_users",
                        "Tweet URL", "Created At", "User Location", "Retweet Count", "Reply Count", "Mention Count",
                        "Longitude", "Latitude", "Replies", "Retweeted Tweet", "Tweet ID", "Profile Image URL"]
        dfgend = dfgend.drop(columns_to_drop, axis=1)
        dfgend['Gender'] = ''

        dfgend = dfgend.drop_duplicates(subset='User Name')

        # Load the model from the output.json file
        model = joblib.load('modelgend.pkl')

        # Predict gender for each tweet
        for index, tweet in dfgend.iterrows():
            features = [tweet['User Name'], tweet['User Description'], tweet['Text']]
            processed_tweet = preprocess_tweet(features, training_columns)
            prediction = predict_gender(model, processed_tweet, training_columns)
            dfgend.at[index, 'Gender'] = prediction

        # Group by gender to get gender distribution
        gender_counts = dfgend.groupby('Gender').size().reset_index(name='Count')

        # Create a pie chart for gender distribution in the corresponding column
        with columns[i]:
            fig = px.pie(gender_counts, values='Count', names='Gender', title='Gender Distribution - ' + file_path)
            st.plotly_chart(fig)


    

##################################################################
with tab3:
    st.header("Data Mining")
    container3=st.container()
    with container3:
       
          
        colta, coltb = st.columns([2, 2])
        with colta:
            
            with st.form(key="taccountform"):
                accounts = st_tags(
                label='# Enter Account:',
                text='Press enter to add more',
                value=[],
                suggestions=[],
                maxtags=4,
                key='1')

                submit = st.form_submit_button(label="Submit")
                if submit:
                    for account in accounts:
                        user = api.get_user(screen_name=account)
                        name = user.name
                        description = user.description


                        # get the list of followers for the user
                        followers = api.get_followers(screen_name=account)
                        follower_list = [follower.screen_name for follower in followers]

                        # get the list of users that the user follows
                        following = api.get_friends(screen_name=account)
                        following_list = [friend.screen_name for friend in following]

                        # find friends that do not follow back
                        not_followed_back = [friend for friend in following_list if friend not in follower_list]

                        # find followers that have not been followed back
                        not_following_back = [follower for follower in follower_list if follower not in following_list]

                        # find friends that follow back
                        followed_back = [friend for friend in following_list if friend in follower_list]

                        # find followers that are also friends
                        following_back = [follower for follower in follower_list if follower in following_list]


                        # get the user's tweets
                        tweets = api.user_timeline(screen_name=account, count=10, tweet_mode='extended')
                        tweets_list = [tweet._json for tweet in tweets]

                        # create a dictionary to store the user's information, tweets, friends, and followers
                        user_data = {
                            'name': name,
                            'description': description,
                            'followers': follower_list,
                            'following': following_list,
                            'not_followed_back': not_followed_back,
                            'not_following_back': not_following_back,
                            'followed_back': followed_back,
                            'following_back': following_back,
                            'tweets': tweets_list
                        }

                       # Create a directory if it doesn't exist
                        os.makedirs("twittl", exist_ok=True)

                        file_path = f"twittl/{account}_data.json"
                        if os.path.exists(file_path):
                            # Load existing data from the file
                            with open(file_path, 'r') as json_file:
                                existing_data = json.load(json_file)

                            # Update the existing data with new data
                            existing_data['name'] = name
                            existing_data['description'] = description
                            existing_data['followers'] = follower_list
                            existing_data['following'] = following_list
                            # Update other fields as needed

                            # Write the updated data back to the file
                            with open(file_path, 'w') as json_file:
                                json.dump(existing_data, json_file)
                        else:
                            # Create a new file and write the data to it
                            user_data = {
                                'name': name,
                                'description': description,
                                'followers': follower_list,
                                'following': following_list,
                                'tweets': tweets_list
                            }
                            # Add other fields as needed

                            with open(file_path, 'w') as json_file:
                                json.dump(user_data, json_file)

         
        with coltb:

            
                
            
            with st.form(key="tkeysform"):
                # Add tag input for keywords
                keywords = st_tags(
                    label='# Enter Keyword(s):',
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
                                'context_annotations', 'text', 'created_at', 'entities', 'source', 'geo', 'public_metrics',
                                'referenced_tweets'
                            ],
                            user_fields=['name', 'username', 'profile_image_url', 'description', 'location'],
                            expansions=[
                                'author_id', 'referenced_tweets.id', 'referenced_tweets.id.author_id',
                                'in_reply_to_user_id', 'entities.mentions.username', 'geo.place_id'
                            ],
                            max_results=20
                        )

                        # Get users list from the includes object
                        users = {u["id"]: u for u in tweets.includes['users']}

                        for tweet in tweets.data:
                            # Initialize 'place_name' and 'in_reply_to_name' variables
                            place_name = ''
                            in_reply_to_name = ''
                            # longitude = None
                            # latitude = None

                            user_id = tweet.author_id
                            if user_id in users:
                                user = users[user_id]
                                user_name = user['name']
                                user_screen_name = user['username']
                                profile_image_url = user['profile_image_url']
                                user_description = user['description']
                                user_location = user.get('location', None)
                                retweeted_user = None  # Initialize the variable

                                # Extract retweet, mention, and reply information
                                retweet_count = tweet.public_metrics['retweet_count']
                                reply_count = tweet.public_metrics['reply_count']
                                mention_count = 0

                                mentioned_users = []

                                if 'entities' in tweet and 'mentions' in tweet.entities:
                                    mentions = tweet.entities['mentions']
                                    mention_count = len(mentions)

                                    for mention in mentions:
                                        mentioned_user_screen_name = mention['username']

                                        mentioned_users.append({
                                            'User Screen Name': mentioned_user_screen_name
                                        })

                                replies = []

                                # Calculate reply count
                                if tweet.in_reply_to_user_id is not None:
                                    reply_count += 1

                                if 'referenced_tweets' in tweet:
                                    referenced_tweets = tweet.referenced_tweets
                                    for referenced_tweet in referenced_tweets:
                                        if referenced_tweet['type'] == 'retweeted':
                                            retweet_count += 1
                                            retweeted_tweet_id = referenced_tweet['id']
                                            try:
                                                retweeted_tweet = client.get_tweet(id=retweeted_tweet_id,
                                                                                tweet_fields=['text', 'author_id'])
                                                if retweeted_tweet:
                                                    retweeted_tweet_data = retweeted_tweet.data
                                                    retweeted_tweet_text = retweeted_tweet_data['text']
                                                    retweeted_tweet_author_id = retweeted_tweet_data['author_id']

                                                    # Get retweeted user's information
                                                    if retweeted_tweet_author_id in users:
                                                        retweeted_user = users[retweeted_tweet_author_id]
                                                        retweeted_user_name = retweeted_user['name']
                                                        retweeted_user_screen_name = retweeted_user['username']

                                                    # Add retweet information to tweet_data
                                                    tweet_data['Retweeted Tweet'] = {
                                                        'Text': retweeted_tweet_text,
                                                        'Author Name': retweeted_user_name,
                                                        'Author Screen Name': retweeted_user_screen_name
                                                    }
                                            except Exception as e:
                                                print(f"Error retrieving retweeted tweet: {e}")

                                        elif referenced_tweet['type'] == 'replied_to':
                                            replied_tweet_id = referenced_tweet['id']
                                            try:
                                                replied_tweet = client.get_tweet(id=replied_tweet_id, tweet_fields=['author_id'])
                                                if replied_tweet:
                                                    replied_user_id = replied_tweet.data['author_id']
                                                    replied_user = users.get(replied_user_id, None)

                                                    if replied_user:
                                                        replied_user_screen_name = replied_user['username']
                                                        replies.append({
                                                            'User Screen Name': replied_user_screen_name
                                                        })
                                            except Exception as e:
                                                print(f"Error retrieving replied tweet: {e}")

                                 # Get latitude and longitude from place_id
                                if 'geo' in tweet and 'place_id' in tweet.geo:
                                    place_id = tweet.geo['place_id']
                                    try:
                                        place = client.get_place(id=place_id, place_fields=['geo'])
                                        if place and 'geo' in place:
                                            latitude = place.geo['coordinates'][0]
                                            longitude = place.geo['coordinates'][1]
                                    except Exception as e:
                                        print(f"Error retrieving place information: {e}")

                                # Add longitude and latitude to tweet_data
                                tweet_data['Longitude'] = longitude if longitude else None
                                tweet_data['Latitude'] = latitude if latitude else None

                                # Get like count and quote count
                                like_count = tweet.public_metrics['like_count']
                                quote_count = tweet.public_metrics['quote_count']

                                if 'entities' in tweet and 'hashtags' in tweet.entities:
                                    hashtags = [tag['tag'] for tag in tweet.entities['hashtags']]
                                else:
                                    hashtags = []

                                full_text = tweet.text

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

                                # Add replies list to tweet_data
                                tweet_data['Replies'] = replies if replies else []

                                results.append(tweet_data)

                        # Create a directory if it doesn't exist
                        os.makedirs("twitkeys", exist_ok=True)

                        # Save the results in a JSON file named with the keyword
                        file_path = os.path.join("twitkeys", f"{keyword}.json")
                        try:
                            if os.path.exists(file_path):
                                # Load existing data from the file
                                with open(file_path, 'r') as json_file:
                                    existing_data = json.load(json_file)

                                # Append new data to the existing data
                                existing_data['data'].extend(results)

                                # Write the combined data back to the file
                                with open(file_path, 'w') as json_file:
                                    json.dump(existing_data, json_file, cls=DateTimeEncoder)
                            else:
                                # Create a new file and write the data to it
                                output = {"data": results}
                                with open(file_path, 'w') as json_file:
                                    json.dump(output, json_file, cls=DateTimeEncoder)
                        except Exception as e:
                            print(f"Error saving results to JSON file: {e}")



        colc, cold=st.columns([2,2])
        with colc:
            container3a=st.container()
            with container3a:

                json_directory = "twittl"

                    # Get the list of JSON files in the directory
                json_files = [file for file in os.listdir(json_directory) if file.endswith(".json")]

                # Loop through each JSON file
                for json_file in json_files:
                    # Open the JSON file and load the data
                    with open(os.path.join(json_directory, json_file), 'r') as f:
                        user_data = json.load(f)

                    # Extract the tweets from the user data
                    tweets_list = user_data['tweets']
                    # user_name = tweet['user']['name']

                    # Create a placeholder for the tweets
                    tweet_placeholder = st.empty()
                    with tweet_placeholder:
                        # Display the tweets
                        # st.subheader(f"Tweets from {json_file[:-10]}")  # Remove the '_data.json' part from the filename
                        for tweet in tweets_list:
                            user_name = tweet['user']['name']
                            full_text = tweet['full_text']
                            
                            # print (user_name)
                            screen_name=tweet['user']['screen_name']
                            # st.write(f"{screen_name}")
                            tweet_content = f"{user_name}: {full_text}"
                            st.write(tweet_content)
                                # st.write("---")
                            time.sleep(0.2)  # Add a delay between each tweet (adjust as needed)
                

        with cold:
            container3b=st.container()
            with container3b:

                def display_tweets(file_path):
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        tweets = data['data']

                    for tweet in tweets:
                        user_name=tweet['User Name']
                        created_at=tweet['Created At']
                        text=tweet['Text']
                        
                        tweet_content = f"{user_name}:{created_at}{text}"
                        st.write(tweet_content)
                        # st.write("---")
                        time.sleep(0.2)  # Add a delay between each tweet (adjust as needed)

                directory = "twitkeys"
                json_files = [file for file in os.listdir(directory) if file.endswith('.json')]

                for json_file in json_files:
                    file_path = os.path.join(directory, json_file)
                    # st.header(json_file[:-5])
                    tweet_placeholder = st.empty()
                    with tweet_placeholder:
                        display_tweets(file_path)


            