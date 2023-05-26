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
# from pyvis.network import Network
from pyvis.network import Network as net
import tweepy
from streamlit_tags import st_tags
from json import JSONEncoder
import PIL
import PIL.Image
import time


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
        # Get a list of files in the folder
        # Specify the directory where the JSON files are located
        folder_path = "twittl"
        files = os.listdir(folder_path)
        num_files = min(4, len(files))  # Limit to first 4 files, or all files if there are fewer than 4

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
        ax.set_title('Top Actors based on Degree Centrality')
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




############################VISUGRAPJH###########################
    colviz1, colviz2, colviz3, colviz4=st.tabs(['Social Network','Degree Centrality', 'Betweenness Centrality', 'Closeness_Centrality'])
    with colviz1:
         visualize_social_network(selected_G)
        
    with colviz2:
        visualize_degree_centrality_network(degree_subgraph, degree_centrality)

    with colviz3:
        visualize_betweenness_centrality_network(betweenness_subgraph, betweenness_centrality)
    
    with colviz4:
        visualize_closeness_centrality_network(selected_G, closeness_centrality)
    
    
##################################################################
with tab3:
    st.header("Data Mining")
    container3=st.container()
    with container3:
        # consumer_key = "wtph1D9eE27h2pTwAfUUZFJGh"
        # consumer_secret = "ueIvSjFVV6MkH7yKtC67ybi6qkPiV4xJun4CsBv8w22lwY6eTF"
        # access_token = "16645853-1WS14NgT2p9m7sMH3s7xU4G5QRN2YBRFXXEYjgEnd"
        # access_token_secret = "Csj5OhyNTUAZOxkBsWi9d7GHnwbQHkLIFgowiBda6lM1o"
        bearer_token = 'AAAAAAAAAAAAAAAAAAAAALMe9wAAAAAAiRJ0vEKHtm8H4w5sW8HRCmjQ6AI%3D8aSrZUbXNvktB7zzus1GIF74g8wfOMlIKn8Obdy7mpgBoIvlXu'

        accounts = []
        auth = tweepy.OAuth1UserHandler('wtph1D9eE27h2pTwAfUUZFJGh', 'ueIvSjFVV6MkH7yKtC67ybi6qkPiV4xJun4CsBv8w22lwY6eTF')
        auth.set_access_token('16645853-1WS14NgT2p9m7sMH3s7xU4G5QRN2YBRFXXEYjgEnd', 'Csj5OhyNTUAZOxkBsWi9d7GHnwbQHkLIFgowiBda6lM1o')
        api = tweepy.API(auth)
        client = tweepy.Client(bearer_token=bearer_token)
          
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
                                'context_annotations', 'text', 'created_at', 'entities', 'source', 'geo', 'public_metrics', 'referenced_tweets'
                            ],
                            user_fields=['name', 'username', 'profile_image_url', 'description', 'location'],
                            expansions=[
                                'author_id', 'referenced_tweets.id', 'referenced_tweets.id.author_id',
                                'in_reply_to_user_id', 'entities.mentions.username', 'geo.place_id'
                            ],
                            max_results=10
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
                                            retweeted_tweet = client.get_tweet(id=retweeted_tweet_id, tweet_fields=['text', 'author_id'])
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

                                        elif referenced_tweet['type'] == 'replied_to':
                                            replied_tweet_id = referenced_tweet['id']
                                            replied_tweet = client.get_tweet(id=replied_tweet_id, tweet_fields=['author_id'])
                                            
                                            if replied_tweet:
                                                replied_user_id = replied_tweet.data['author_id']
                                                replied_user = users.get(replied_user_id, None)
                                                
                                                if replied_user:
                                                    replied_user_screen_name = replied_user['username']
                                                    replies.append({
                                                        'User Screen Name': replied_user_screen_name
                                                    })
                                # Add replies list to tweet_data
                                tweet_data['Replies'] = replies if replies else []                                   
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
                                
                                results.append(tweet_data)
                                
                            # Create a directory if it doesn't exist
                            os.makedirs("twitkeys", exist_ok=True)
                            
                            # Save the results in a JSON file named with the keyword
                            output = {"data": results}
                            # Save the results in a JSON file named with the keyword
                            file_path = os.path.join("twitkeys", f"{keyword}.json")
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


            