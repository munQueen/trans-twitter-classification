# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:08:41 2019

@author: jwulz
"""
from twitterscraper import query_tweets
import pandas as pd
import datetime
import os

# set the search term here:
query_term = 'cisgender'

tweets = query_tweets(query="cisgender", begindate=datetime.date(2019, 6, 1), enddate=datetime.date(2019, 10, 1), limit=2000, lang='en')
tweet_list = (t.__dict__ for t in tweets)
tweet_df = pd.DataFrame(tweet_list)
filename = query_term + ".csv"


os.chdir('C:/Learning/ML/trans_twitter/csv_files')
language_regex = 'lang="en"'


# use regular expressions to remove URLS and images 
tweet_df = tweet_df[tweet_df.html.str.contains(language_regex)]
tweet_df.text.replace('http\S+', '', regex=True, inplace=True)
tweet_df.text.replace('pic.twitter\S+', '', regex=True, inplace=True)
tweet_df = tweet_df.drop_duplicates(subset='text')

tweet_df.to_csv(filename)