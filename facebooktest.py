import snscrape.modules.facebook as snfb
import pandas as pd

query = "PSI"
fbpost = []
limit = 5000


for post in snfb.FacebookPost(query).get_items():

    print(vars(post))
    # break
#     if len(post) == limit:
#         break
#     else:
#         post.append([])

# df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
# print(df)

# to save to csv
# df.to_csv('tweets.csv')
