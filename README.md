# Airline-tweets-sentiments
## Problem Description
Given dataset contains data of tweets on various airline’s twitter handles.

It contains a total of 12 columns, out of which one column specifies the sentiment of the tweet. All other columns provide various information related to what was the tweet, where was it posted from, when was it posted, it's retweeted; etc.

My task was to build a machine learning / deep learning model to predict the sentiment of the tweet using all or some of the other given columns

## Data Description
Description of columns of the dataset is given below -

tweet_id -- Id of the tweet

airline_sentiment -- Sentiment of the tweet (Target variable)

airline_sentiment_confidence -- Confidence with which the given sentiment was determined

negativereason_confidence -- Confidence with which the negative reason of tweet was predicted

name -- Name of the person who tweeted

retweet_count -- Number of retweets

text -- Text of the tweet whose sentiment has to be predicted

tweet_created -- Time at which the tweet was created

tweet_location -- Location from where the tweet was posted

user_timezone -- Time zone from where the tweet was posted

negativereason -- Reason for which user posted a negative tweet

airline -- Airline for which the tweet was posted

## Inspiration
The data is a nice combination of Numeric and Non-numeric featutres. it can be used for sentiment analysis.

## content
1. Introduction
2. Data Injection
3. Data Visualisation
4. Preprocessing
5. Training                                                                                                                            
5.1. Logistic Regression                                                                                                  
5.2.Artificial Neural Network
6. Evaluation with graph
