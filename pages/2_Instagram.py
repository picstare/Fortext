import streamlit as st
from instaloader import Instaloader, Profile
import json
import os

# Streamlit app title
st.title('Instagram User Search')

# Streamlit form to input username
username = st.text_input('Enter Instagram username')

# Streamlit button to submit the form
if st.button('Search'):
    try:
        # Create Instaloader instance
        loader = Instaloader()

        # Your Instagram credentials
      

        # Login using the credentials
        

        # Retrieve user profile
        profile = Profile.from_username(loader.context, username)

        # Display user data
        st.subheader('User Data')
        st.write('Full Name:', profile.full_name)
        st.write('Username:', profile.username)
        st.write('Bio:', profile.biography)
        st.write('Followers:', profile.followers)
        st.write('Following:', profile.followees)
        st.write('Posts:', profile.mediacount)
        st.write('Profile Picture URL:', profile.profile_pic_url)

        # Create directory if it doesn't exist
        save_dir = os.path.join(os.getcwd(), 'insperson')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create dictionary for user and post data
        user_data = {
            'Full Name': profile.full_name,
            'Username': profile.username,
            'Bio': profile.biography,
            'Followers': profile.followers,
            'Following': profile.followees,
            'Posts': profile.mediacount,
            'Profile Picture URL': profile.profile_pic_url,
            'Posts Data': []
        }

        # Retrieve and save post data
        for post in profile.get_posts():
            # Create dictionary for post data
            post_data = {
                'Image URL': post.url,
                'Caption': post.caption,
                'Likes': post.likes,
                'Comments': post.comments,
                'Posted on': post.date_local
            }

            # Add post data to user data
            user_data['Posts Data'].append(post_data)

            # Display post data
            st.image(post.url)
            st.write('Caption:', post.caption)
            st.write('Likes:', post.likes)
            st.write('Comments:', post.comments)
            st.write('Posted on:', post.date_local)
            st.write('---')

        # Save user and post data to JSON file
        json_file_path = os.path.join(save_dir, f'{username}_data.json')
        with open(json_file_path, 'w') as f:
            json.dump(user_data, f, indent=4)

    except Exception as e:
        st.error('An error occurred: ' + str(e))
