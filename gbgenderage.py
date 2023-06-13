import os
import json
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import streamlit as st
import numpy as np
from sklearn.metrics import accuracy_score

st.write('streamlit')


# def preprocess_tweet(tweet, training_columns):
#     processed_tweet = {
#         'User Name': str(tweet[0]),
#         'User Description': str(tweet[1]).lower() if tweet[1] else '',
#         'Text': str(tweet[2]).lower(),
#     }

#     processed_tweet = {k: float(v) if isinstance(v, str) and v.isnumeric() else v for k, v in processed_tweet.items()}
#     processed_tweet = pd.DataFrame(processed_tweet, index=[0])

#     # Perform one-hot encoding on the categorical variables
#     processed_tweet_encoded = pd.get_dummies(processed_tweet)
#     processed_tweet_encoded = processed_tweet_encoded.reindex(columns=training_columns, fill_value=0)

#     return processed_tweet_encoded.values.flatten()


# def predict_gender(model, features, training_columns):
#     processed_tweet = preprocess_tweet(features, training_columns)
#     prediction = model.predict([processed_tweet])
#     return prediction[0]



# dfout = pd.read_json('output.json')

# st.dataframe(dfout)
# print ("DFOUT:",dfout)

# # Prepare the data
# if 'Gender' in dfout.columns:
#     X = dfout.drop('Gender', axis=1)
# else:
#     X = dfout.copy()

# if 'Gender' in dfout.columns:
#     y = dfout['Gender']
# else:
#     # Handle the case when 'Gender' column is missing
#     # For example, you can print an error message or take appropriate action
#     st.write('Gender not in df.columns')

# # Perform one-hot encoding on the categorical variables in X
# X_encoded = pd.get_dummies(X)

# # Get the training columns from the X_encoded DataFrame
# training_columns = X_encoded.columns

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# # Create a Gradient Boosting Classifier model
# model = GradientBoostingClassifier()

# # Fit the model to the training data
# model.fit(X_train, y_train)

# # Predict the gender for the test data
# y_pred = model.predict(X_test)

# # Evaluate the model's performance
# accuracy = accuracy_score(y_test, y_pred)
# st.write('Accuracy:', accuracy)

# # Save the trained model to a file
# joblib.dump(model, 'modelgend.pkl')

# # Get the file paths of all JSON files in the "twitkeys" folder
# file_paths = glob.glob('twitkeys/*.json')

# # Sort the file paths by modification time (newest to oldest)
# file_paths.sort(key=os.path.getmtime, reverse=True)

# # Select the four newest files
# default_files = file_paths[:1]

# # Allow users to select multiple files using a multiselect widget
# selected_files = st.multiselect("Select JSON Files", file_paths, default=default_files, key='gensel')

# data_list = []

# for file_path in selected_files:
#     with open(file_path, 'r') as file:
#         json_data = json.load(file)
#         data_list.extend(json_data["data"])

# dfgend = pd.DataFrame(data_list)
# # Drop irrelevant columns
# columns_to_drop = ["User Screen Name", "User Location", "Hashtags", "Source", "in_reply_to_name", "mentioned_users",
#                 "Tweet URL", "Created At", "User Location", "Retweet Count", "Reply Count", "Mention Count",
#                 "Longitude", "Latitude", "Replies", "Retweeted Tweet", "Tweet ID", "Profile Image URL"]
# dfgend = dfgend.drop(columns_to_drop, axis=1)
# dfgend['Gender'] = ''

# dfgend = dfgend.drop_duplicates(subset='User Name')

# st.title("Gender Prediction from Twitter Data")


# # Load the model from the output.json file
# model = joblib.load('modelgend.pkl')


# for index, tweet in dfgend.iterrows():
#     features = [tweet['User Name'], tweet['User Description'], tweet['Text']]
#     processed_tweet = preprocess_tweet(features, training_columns)
#     prediction = predict_gender(model, processed_tweet, training_columns)
#     dfgend.at[index, 'Gender'] = prediction



# st.title("Predicted Gender")
# st.dataframe(dfgend)




data_list = []

folder_path = "twitkeys"

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            data_list.extend(json_data["data"])

df = pd.DataFrame(data_list)
# Drop irrelevant columns
columns_to_drop = ["User Screen Name", "User Location", "Hashtags", "Source", "in_reply_to_name", "mentioned_users", "Tweet URL",
                   "Created At", "User Location", "Retweet Count", "Reply Count", "Mention Count", "Longitude", "Latitude", "Replies", "Retweeted Tweet", "Tweet ID", "Profile Image URL"]
df = df.drop(columns_to_drop, axis=1)



df['Gender'] = ''
df = df.drop_duplicates(subset='User Name')

st.dataframe(df)
print (df)
editeddf=st.experimental_data_editor(df)

editeddf.replace(['NA', 'N/A', 'None', ''], pd.NA, inplace=True)
editeddf.dropna(subset=['Gender'], inplace=True)
st.dataframe(editeddf)

editeddf.to_json('output.json', orient='records')

# Load the data from the JSON file into a DataFrame


# # ###############################################
# # df = pd.read_json('output1.json')

# # # Prepare the data
# # if 'Gender' in df.columns:
# #     X = df.drop('Gender', axis=1)
# # else:
# #     X = df.copy()

# # if 'Gender' in df.columns:
# #     y = df['Gender']
# # else:
# #     # Handle the case when 'Gender' column is missing
# #     # For example, you can print an error message or take appropriate action
# #     st.write('Gender not in df.columns')


# # # Perform one-hot encoding on the categorical variables in X
# # X_encoded = pd.get_dummies(X)

# # # Get the training columns from the X_encoded DataFrame
# # training_columns = X_encoded.columns

# # # Split the data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# # # Create a Gradient Boosting Classifier model
# # model = GradientBoostingClassifier()

# # # Fit the model to the training data
# # model.fit(X_train, y_train)

# # # Predict the gender for the test data
# # y_pred = model.predict(X_test)

# # # Evaluate the model's performance
# # accuracy = accuracy_score(y_test, y_pred)
# # st.write('Accuracy:', accuracy)

# # # Save the trained model to a file
# # joblib.dump(model, 'modelgend.pkl') 

# # ############################################### 

# # def preprocess_tweet(tweet, training_columns):
# #         if isinstance(tweet, dict):
# #             processed_tweet = {
# #                 'User Name': str(tweet['User Name']),
# #                 'User Description': str(tweet['User Description']).lower() if tweet['User Description'] else '',
# #                 'Text': str(tweet['Text']).lower(),
# #             }
# #         elif isinstance(tweet, list) and len(tweet) > 0:
# #             processed_tweet = {
# #                 'User Name': str(tweet[0]['User Name']),
# #                 'User Description': str(tweet[0]['User Description']).lower() if tweet[0]['User Description'] else '',
# #                 'Text': str(tweet[0]['Text']).lower(),
# #             }
# #         else:
# #             processed_tweet = {
# #                 'User Name': '',
# #                 'User Description': '',
# #                 'Text': '',
# #             }

# #         processed_tweet = {k: float(v) if isinstance(v, str) and v.isnumeric() else v for k, v in processed_tweet.items()}
# #         processed_tweet = pd.DataFrame(processed_tweet, index=[0])

# #         # Perform one-hot encoding on the categorical variables
# #         processed_tweet_encoded = pd.get_dummies(processed_tweet)
# #         processed_tweet_encoded = processed_tweet_encoded.reindex(columns=training_columns, fill_value=0)

# #         return processed_tweet_encoded.values.flatten()


# # def predict_gender(model, features, training_columns):
# #     processed_tweet = preprocess_tweet(features, training_columns)
# #     prediction = model.predict([processed_tweet])
# #     return prediction[0]


# # # Get the file paths of all JSON files in the "twitkeys" folder
# # file_paths = glob.glob('twitkeys/*.json')

# # # Sort the file paths by modification time (newest to oldest)
# # file_paths.sort(key=os.path.getmtime, reverse=True)

# # # Select the four newest files
# # default_files = file_paths[:1]

# # # Allow users to select multiple files using a multiselect widget
# # selected_files = st.multiselect("Select JSON Files", file_paths, default=default_files, key='gensel')

# # data_list = []

# # for file_path in selected_files:
# #     with open(file_path, 'r') as file:
# #         json_data = json.load(file)
# #         data_list.extend(json_data["data"])

# # df = pd.DataFrame(data_list)
# # # Drop irrelevant columns
# # columns_to_drop = ["User Screen Name", "User Location", "Hashtags", "Source", "in_reply_to_name", "mentioned_users",
# #                 "Tweet URL", "Created At", "User Location", "Retweet Count", "Reply Count", "Mention Count",
# #                 "Longitude", "Latitude", "Replies", "Retweeted Tweet", "Tweet ID", "Profile Image URL"]
# # df = df.drop(columns_to_drop, axis=1)
# # df['Gender'] = ''

# # df = df.drop_duplicates(subset='User Name')

# # st.title("Gender Prediction from Twitter Data")
# # st.dataframe(df)

# # # Load the model from the output.json file
# # model = joblib.load('modelgend.pkl')

# # for _, tweet in df.iterrows():
# #     features = [tweet['User Name'], tweet['User Description'], tweet['Text']]
# #     processed_tweet = preprocess_tweet(features, training_columns)
# #     prediction = predict_gender(model, processed_tweet, training_columns)
# #     df.loc[df['User Name'] == tweet['User Name'], 'Gender'] = prediction

# # # Iterate over the tweet data and make predictions
# # df['Gender'] = predict_gender(model, df[['User Name', 'User Description', 'Text']])

# # st.title("Predicted Gender")
# # st.dataframe(df)
