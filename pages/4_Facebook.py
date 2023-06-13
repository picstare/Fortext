import os
import json
from datetime import datetime
from json import JSONEncoder
import streamlit as st
from facebook_scraper import get_posts

# Custom JSON encoder to handle non-serializable objects
class CustomJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)

# Streamlit app title
st.title("Facebook Page Scraper")

# Get user input for page URL and cookies
page_url = st.text_input("Enter Facebook Page URL", help="Enter the URL of the Facebook page you want to scrape")
c_user = st.text_input("Enter c_user Cookie", help="Enter the value of the c_user cookie")
xs = st.text_input("Enter xs Cookie", help="Enter the value of the xs cookie")

# Define a button to start scraping
start_scraping = st.button("Start Scraping")

# Check if button is clicked and page URL and cookies are provided
if start_scraping and page_url and c_user and xs:
    # Format the cookies
    cookies = {
        "c_user": c_user,
        "xs": xs
    }

    # Scrape the page using cookies
    st.info("Scraping posts using provided cookies...")

    # Create a folder for saving the JSON files
    folder_name = "fbpages"
    os.makedirs(folder_name, exist_ok=True)

    # Scraping and save results in JSON file
    file_name = f"{folder_name}/{page_url.split('/')[-1]}.json"
    scraped_posts = []
    for post in get_posts(page_url, cookies=cookies, pages=3, extra_info=True, options={"comments": True}):
        scraped_posts.append(post)

    # Save the results in a JSON file with custom encoder
    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump(scraped_posts, json_file, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)

    st.success(f"Scraping complete. Results saved in {file_name}.")

# Display a message if button is clicked but page URL or cookies are missing
elif start_scraping and (not page_url or not c_user or not xs):
    st.warning("Please enter the URL of the Facebook page and both c_user and xs cookies to start scraping.")



# import streamlit as st
# from facebook_scraper import get_posts
# import http.cookiejar

# # Streamlit app title
# st.title("Facebook Page Scraper")

# # Get user input for page URL
# page_url = st.text_input("Enter Facebook Page URL", help="Enter the URL of the Facebook page you want to scrape")

# # Define a button to start scraping
# start_scraping = st.button("Start Scraping")

# # Check if button is clicked and page URL is provided
# if start_scraping and page_url:
#     # Retrieve cookies from the browser
#     cookie_jar = http.cookiejar.CookieJar()
#     cookie_processor = urllib.request.HTTPCookieProcessor(cookie_jar)
#     opener = urllib.request.build_opener(cookie_processor)
#     opener.open("https://www.facebook.com")
#     cookies = []
#     for cookie in cookie_jar:
#         cookies.append(f"{cookie.name}={cookie.value}")

#     # Scrape the page using cookies
#     st.info("Scraping posts using retrieved cookies...")
#     for post in get_posts(page_url, cookies=cookies, pages=5):
#         st.write("Post Date:", post['time'])
#         st.write("Post Text:", post['text'])
#         st.write("---")

# # Display a message if button is clicked but page URL is missing
# elif start_scraping and not page_url:
#     st.warning("Please enter the URL of the Facebook page to start scraping.")