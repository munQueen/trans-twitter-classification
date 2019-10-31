# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:13:04 2019

@author: jwulz
"""
import glob
import pandas as pd
import os
import re
import numpy as np
import nltk
#import spacy
from sklearn.metrics import confusion_matrix, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from nltk import TweetTokenizer
from nltk.stem import WordNetLemmatizer 
import pickle
from scipy import sparse

os.chdir('C:/Learning/ML/trans_twitter/labeled_data')



#open up all of the labeled CSV files, take all of the labeled data and consolidate into one dataframe
datafiles = [f for f in glob.glob("*.csv")]
collected_data = pd.DataFrame(columns=['text', 'score'])
for file in datafiles:
   print("on:", file)
   csv = pd.read_csv(file)
   df = csv[(csv.irrelevant != 1) & (csv.score >= 0) & (csv.score <= 5)][['text', 'score']]
   collected_data = collected_data.append(df)

collected_data = collected_data.reset_index().drop(labels='index', axis=1)
collected_data.score.value_counts()


os.chdir('C:/Learning/ML/trans_twitter/github_data')
collected_data.to_csv('collected_data.csv')


#two different possible input methods:
#1. binary (only positive and negative, no concept of neutral)
#2. ternary (positive/neutral/negative)
full_data = collected_data
full_data['binary_category'] = 'negative'
full_data.loc[full_data.score >= 3, 'binary_category'] = 'supportive'
full_data['ternary_category'] = 'negative'
full_data.loc[full_data.score == 3, 'ternary_category'] = 'neutral'
full_data.loc[full_data.score >= 4, 'ternary_category'] = 'supportive'

full_data.binary_category.value_counts()
full_data.ternary_category.value_counts()





#VADER analysis
analyzer = SentimentIntensityAnalyzer()
vader_scores = full_data.text.apply(analyzer.polarity_scores).apply(pd.Series)
full_data = full_data.merge(vader_scores, how='left', left_index=True, right_index=True)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.kdeplot(full_data.loc[full_data['ternary_category'] == 'negative', 'compound'], color='red', label='Negative')
sns.kdeplot(full_data.loc[full_data['ternary_category'] == 'neutral', 'compound'], color='grey', label='Neutral')
sns.kdeplot(full_data.loc[full_data['ternary_category'] == 'supportive', 'compound'], color='deepskyblue', label='Supportive')
plt.xlabel('VADER sentiment score')
plt.ylabel('Kernel Density Estimate')
plt.show()




extreme_supportive_tweets= full_data.loc[(full_data['compound'] < -0.8) & (full_data.ternary_category == 'supportive')]
extreme_negative_tweets = full_data.loc[(full_data['compound'] > 0.8) & (full_data.ternary_category == 'negative')]   
   
   
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
     




ttokenizer = TweetTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
#we want to keep ARE and AREN'T as these are crucial in trans women are women etc
stop_words.remove("are")
stop_words.remove("aren't")
stop_words.remove("not")


  
lemmatizer = WordNetLemmatizer() 
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
full_data['cleaned_text']= normalize_corpus(full_data.text)


def text_preprocessing_noquotes(doc):
   doc = re.sub(r'[^a-zA-Z#\s]', '', doc, re.I|re.A)
   doc = doc.lower()
   doc = doc.strip()
   
   tokens = ttokenizer.tokenize(doc)
   tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
   doc = ' '.join(tokens)
   
   return(doc)
   
normalize_corpus_noquotes = np.vectorize(text_preprocessing_noquotes)
full_data['cleaned_text_noquotes']= normalize_corpus_noquotes(full_data.text)




#look for certain phrases 
twaw = 'trans women are women' in full_data.cleaned_text 
full_data['twaw_flag'] = False
full_data['tmam_flag'] = False
for index, row in full_data.iterrows():
   if 'trans women are women' in row['cleaned_text']:
      full_data.loc[index, 'twaw_flag'] = True
   if 'trans men are men' in row['cleaned_text']:
      full_data.loc[index, 'tmam_flag'] = True
      

#sample down the data to have balanced classes 
v_counts = full_data.binary_category.value_counts()
negative_count = v_counts['negative']
positive_count = v_counts['supportive']

#downsampling
#supportive_sample_binary = full_data[full_data.binary_category == 'supportive'].sample(n=negative_count, random_state=42)
#binary_data = full_data[full_data.binary_category == 'negative'].append(supportive_sample_binary)
#
#supportive_sample_ternary = full_data[full_data.ternary_category == 'supportive'].sample(n=negative_count, random_state=42)
#ternary_data = full_data[full_data.ternary_category == 'negative'].append(supportive_sample_ternary)




#logistic regression using only VADER scores
#from sklearn.linear_model import LogisticRegression
#X_train, X_test, y_train, y_test = train_test_split(binary_data[['compound']], binary_data['binary_category'], test_size=0.4, random_state=42)
#log_reg = LogisticRegression()
#log_reg.fit(X_train, y_train)
#vader_pred = log_reg.predict(X_test)
#print("VADER logreg cm:", confusion_matrix(y_test, vader_pred))
#print("VADER accuracy:", accuracy_score(vader_pred, y_test))


#TFIDF - no quotes 
tfidf_noquotes = TfidfVectorizer(analyzer='word', token_pattern=r"\"[^\"]+\"|[\w]+", ngram_range=(1,4))
tf_scores_noquotes = tfidf_noquotes.fit_transform(binary_data.cleaned_text_noquotes)
print('#of TFIDF features, no quotes:', len(tfidf_noquotes.get_feature_names()))
X_train, X_test = train_test_split(full_data, test_size=0.4, random_state=42)
upsample_amount = X_train.binary_category.value_counts()['supportive'] - X_train.binary_category.value_counts()['negative'] 
upsample_data = X_train[X_train.binary_category == 'negative'].sample(n=upsample_amount, random_state=42, replace=True)
X_train = X_train.append(upsample_data)
y_train = X_train.binary_category
y_test = X_test.binary_category
X_train = tfidf_noquotes.transform(X_train.cleaned_text)
X_test = tfidf_noquotes.transform(X_test.cleaned_text)


MNB = MultinomialNB()
MNB.fit(X_train, y_train)
print("Accuracy on train:", accuracy_score(MNB.predict(X_train), y_train))
print("Accuracy on test:", accuracy_score(MNB.predict(X_test), y_test))



cr = classification_report(y_test, MNB.predict(X_test))
print(cr)
cm = confusion_matrix(y_test, MNB.predict(X_test), labels=['supportive', 'negative'])
print(cm)



#find a list of the most common terms

#looking at individual features
words = np.array(tfidf_noquotes.get_feature_names())
x = sparse.eye(X_test.shape[1])
probs = MNB.predict_log_proba(x)[:, 0]
ind = np.argsort(probs)
good_words = words[ind[:25]]
bad_words = words[ind[-25:]]

good_prob = probs[ind[:25]]
bad_prob = probs[ind[-25:]]

print("Good words\t     P")
for w, p in zip(good_words, good_prob):
   print("{:>20}".format(w), "{:.2f}".format(1-np.exp(p)))
   
print("Bad words\t     P")
for w, p in zip(bad_words, bad_prob):
   print("{:>20}".format(w), "{:.2f}".format(1-np.exp(p)))
   




#TFIDF - quotes- upsampling
tfidf = TfidfVectorizer(analyzer='word', token_pattern=r"\"[^\"]+\"|[\w]+", ngram_range=(1,4))
tf_scores = tfidf.fit_transform(full_data.cleaned_text)
X_train, X_test = train_test_split(full_data, test_size=0.4, random_state=42)

#upsample the training data to have balanced classes
upsample_amount = X_train.binary_category.value_counts()['supportive'] - X_train.binary_category.value_counts()['negative'] 
upsample_data = X_train[X_train.binary_category == 'negative'].sample(n=upsample_amount, random_state=42, replace=True)
X_train = X_train.append(upsample_data)
y_train = X_train.binary_category
y_test = X_test.binary_category
X_train = tfidf.transform(X_train.cleaned_text)
X_test = tfidf.transform(X_test.cleaned_text)

print('#of TFIDF features, with quotes:', len(tfidf.get_feature_names()))
MNB = MultinomialNB()
MNB.fit(X_train, y_train)
print("Accuracy on train:", accuracy_score(MNB.predict(X_train), y_train))
print("Accuracy on test:", accuracy_score(MNB.predict(X_test), y_test))

cr = classification_report(y_test, MNB.predict(X_test))
print(cr)
cm = confusion_matrix(y_test, MNB.predict(X_test), labels=['supportive', 'negative'])
print(cm)


#auc = roc_curve(y_test.to_numpy(), MNB.predict(X_test), pos_label='supportive')
#fpr, tpr, thresholds = roc_curve(y_test, MNB.predict(X_test), pos_label='supportive')



os.chdir('C:/Learning/ML/trans_twitter/scikit_models')
pickle.dump(MNB, open("MNB.p", "wb"))
pickle.dump(tfidf, open('tfidf.p', 'wb'))






#two part model with the vader scores 
#find the vader scores for the training data
vaders_train = full_data.loc[y_train.index, 'compound']
vaders_test  = full_data.loc[y_test.index,  'compound']
pred_prob_train = MNB.predict_proba(X_train)
pred_prob_test  = MNB.predict_proba(X_test)

model_input_train = pd.DataFrame({'vader_scores':vaders_train, 'naive_bayes_probs':pred_prob_train[:,1]})
model_input_test = pd.DataFrame({'vader_scores':vaders_test, 'naive_bayes_probs':pred_prob_test[:,1]})

logreg = LogisticRegression()
logreg.fit(model_input_train, y_train)
final_preds_train = logreg.predict(model_input_test)

print("Accuracy on train:", accuracy_score(logreg.predict(model_input_train), y_train))
print("Accuracy on test:", accuracy_score(final_preds_train, y_test))

cr = classification_report(y_test, final_preds_train)
cm = confusion_matrix(y_test, final_preds_train, labels=['supportive', 'negative'])
print(cm)



false_positives = (MNB.predict(X_test) != y_test) & (y_test == 'negative')
false_positives = false_positives.rename("false_positives")

false_negatives = (MNB.predict(X_test) != y_test) & (y_test == 'supportive')
false_negatives = false_negatives.rename("false_negatives")

full_data = full_data.merge(false_positives, how='left', left_index=True, right_index=True)
full_data = full_data.merge(false_negatives, how='left', left_index=True, right_index=True)

fp = full_data[full_data.false_positives == True]
fn = full_data[full_data.false_negatives == True]


#look at individual features of tfidf
words = np.array(tfidf.get_feature_names())
x = sparse.eye(X_test.shape[1])
probs = MNB.predict_log_proba(x)[:, 0]
ind = np.argsort(probs)
good_words = words[ind[:10]]
bad_words = words[ind[-10:]]

good_prob = probs[ind[:10]]
bad_prob = probs[ind[-10:]]

print("Good words\t     P")
for w, p in zip(good_words, good_prob):
   print("{:>20}".format(w), "{:.3f}".format(1-np.exp(p)))
   
print("Bad words\t     P")
for w, p in zip(bad_words, bad_prob):
   print("{:>20}".format(w), "{:.3f}".format(1-np.exp(p)))
   


#code to examine a single tweet
words = np.array(tfidf.get_feature_names())
x = sparse.eye(X_test.shape[1])
probs = MNB.predict_log_proba(x)[:, 0]
tweet_tfidf = tfidf.transform(["god zuko is such a trans girl"])
tweet_ft_inds = tweet_tfidf.nonzero()[1]
tweet_probs = probs[tweet_ft_inds]
1-np.exp(tweet_probs)

for i in range(len(tweet_probs)):
   print(words[tweet_ft_inds[i]], "{:.3f}".format(1-np.exp(tweet_probs[i])))

