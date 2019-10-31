# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:23:01 2019

@author: jwulz
"""

from twitterscraper import query_tweets
import pandas as pd
import datetime
import os
import re
import nltk
from nltk import TweetTokenizer
from nltk.stem import WordNetLemmatizer 
import numpy as np
import pickle

#read in the list of terms

os.chdir('C:/Learning/ML/trans_twitter')
terms = pd.read_csv('trans_related_terms.csv')
term_list = terms.term

#fetch tweets
tweets = query_tweets(query="from:marlow_f", begindate=datetime.date(2018, 1, 1), enddate=datetime.date(2019, 10, 1), limit=2000, lang='en')
tweet_list = (t.__dict__ for t in tweets)
tweet_df = pd.DataFrame(tweet_list)
tweet_df = tweet_df.drop_duplicates()



#process text
ttokenizer = TweetTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer() 

def text_preprocessing_noquotes(doc):
   doc = re.sub(r'[^a-zA-Z#\s]', '', doc, re.I|re.A)
   doc = doc.lower()
   doc = doc.strip()
   tokens = ttokenizer.tokenize(doc)
   tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
   doc = ' '.join(tokens)
   return(doc)
normalize_corpus_noquotes = np.vectorize(text_preprocessing_noquotes)
tweet_df['cleaned_text_noquotes']= normalize_corpus_noquotes(tweet_df.text)


def process_quotes(text):
   '''this function finds quotation marks in a text and puts each word in a quote in individual quotation marks'''
   text = text.strip()
   quote_count = text.count('"')
   
   if quote_count == 0:
      return(text)
   
   #odd number of quotes- too complex to interpret, just remove all quotes
   elif (quote_count % 2) == 1:
      text = re.sub(r'[^a-zA-Z#\s]', '', text, re.I|re.A)
      return(text)

   
   #even number of quotes: needs some logic
   else:
      split_text = text.split('"')      
      assembled_string = ''
      #if the string starts with a quoted word:
      if text[0] == '"':
         #handle the first string with quotes, then remove it from split_text and continue to the normal pattern 
         split_text.pop(0)
         for word in split_text[0].split(' '):
            if word != '':
               assembled_string = assembled_string + '"' + word + '" '
         assembled_string = assembled_string + ' '
         split_text.pop(0)
                  
      #need to take every 2nd string, place quotes around every word in those strings 
      i = 0 
      while i < len(split_text) - 1:
         assembled_string = assembled_string + split_text[i]
         for word in split_text[i+1].split(' '):
            if word != '':
               assembled_string = assembled_string + '"' + word + '" '
         i = i + 2
                        
      #if the string ends with a quoted word, we are done
      #but if the string doesn't end with a quote, we need to add the last section      
      if text[len(text) - 1] != '"':
         assembled_string = assembled_string + split_text[len(split_text) - 1]
         
      #remove extra whitespace    
      assembled_string = " ".join(assembled_string.split())
      return(assembled_string)
def text_preprocessing(doc):
   doc = re.sub(r'[“”]', '"', doc, re.I|re.A)
   doc = re.sub(r'[^a-zA-Z"#\s]', '', doc, re.I|re.A)
   doc = doc.lower()
   doc = doc.strip()
   tokens = ttokenizer.tokenize(doc)
   tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
   doc = ' '.join(tokens)
   
   doc = process_quotes(doc)
   
   return(doc)
   
normalize_corpus = np.vectorize(text_preprocessing)
tweet_df['cleaned_text']= normalize_corpus(tweet_df.text)

#for each tweet, label each tweet if it features any of the words 
tweet_df['trans_related'] = False
for index, row in tweet_df.iterrows():
   if any(term in row['cleaned_text_noquotes'] for term in term_list):
      tweet_df.loc[index, 'trans_related'] = True
      
   
trans_tweets = tweet_df.loc[tweet_df['trans_related'] == True]

#run the trained Naive Bayes on these tweets to create predictions
#NOTE - need to figure out how to save the vectorizer and model and load them here

os.chdir('C:/Learning/ML/trans_twitter/scikit_models')
MNB_p = pickle.load(open("MNB.p", 'rb'))
tfidf_p = pickle.load(open("tfidf.p", 'rb'))
tfidf_tweets = tfidf_p.transform(trans_tweets.cleaned_text)
predictions = MNB_p.predict(tfidf_tweets)
predict_probs = MNB_p.predict_proba(tfidf_tweets)
trans_tweets.loc[:,'predictions'] = predictions
trans_tweets.loc[:,'predict_probs'] = predict_probs[:,1]


#identify the most strongly positive, strongly negative, and middling tweets
tweet_count = tweet_df.shape[0]
trans_tweet_count = trans_tweets.shape[0]
trans_tweet_vc = trans_tweets.predictions.value_counts()
if 'supportive' in trans_tweet_vc.index:
   supportive_tweet_count = trans_tweets.predictions.value_counts()['supportive']
else:
   supportive_tweet_count = 0
if 'negative' in trans_tweet_vc.index:
   negative_tweet_count = trans_tweets.predictions.value_counts()['negative'] 
else:
   negative_tweet_count = 0


supportive_tweet_pct = supportive_tweet_count / trans_tweet_count
supportive_sample = trans_tweets.loc[trans_tweets.predictions == 'supportive', 'text'].sample(n=min(5, supportive_tweet_count))
negative_sample = trans_tweets.loc[trans_tweets.predictions == 'negative', 'text'].sample(n=min(5, negative_tweet_count))