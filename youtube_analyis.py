# -*- coding: utf-8 -*-
"""Youtube-Analyis.ipynb

Original file is located at
    https://colab.research.google.com/drive/1QwrvhqRV24SRouAkT4dCZ3gM_xCn-D22

# Importing the youtube comments for a particular video using Youtube API
"""

import requests
import csv
import pandas as pd
import os
from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

# Set up the YouTube Data API key
api_key = ""  # Replace with your own API key

# Create a YouTube Data API client
youtube = build("youtube", "v3", developerKey=api_key)

# Define the video ID for the video you want to fetch comments from
video_id = "2Vv-BfVoq4g"

# Function to fetch comments and save them to a CSV file
def fetch_comments_to_csv(api_key, video_id, max_results=1000):
    # Create a CSV file for writing
    csv_filename = "youtube_comments.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header row
        csv_writer.writerow(["Author", "Comment", "Likes"])

        # Initialize variables for pagination
        page_token = None
        total_comments = 0

        while total_comments < max_results:
            # Fetch comments using the YouTube Data API
            comments = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - total_comments),
                textFormat="plainText",
                pageToken=page_token
            ).execute()

            # Process and write comments to the CSV file
            for comment in comments.get("items", []):
                snippet = comment["snippet"]["topLevelComment"]["snippet"]
                author_name = snippet["authorDisplayName"]
                comment_text = snippet["textDisplay"]
                like_count = snippet["likeCount"]
                csv_writer.writerow([author_name, comment_text, like_count])
                total_comments += 1

            # Check if there are more comments to fetch
            if "nextPageToken" in comments:
                page_token = comments["nextPageToken"]
            else:
                break

# Fetch and save comments
fetch_comments_to_csv(api_key, video_id, max_results=1000)
print('Comments saved to youtube_comments.csv')

"""# Sentimental Analysis on saved comments"""

# Read the CSV file
df = pd.read_csv('youtube_comments.csv')

# Display the first few rows of the DataFrame
df.head(40)

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(Comment):
    # Tokenize the tweet into sentences
    sentences = tokenize.sent_tokenize(Comment)

    num_positive = 0
    num_negative = 0
    num_neutral = 0

    # Analyze sentiment for each sentence
    for sentence in sentences:
        sentiment_scores = sia.polarity_scores(sentence)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            num_positive += 1
        elif compound_score <= -0.05:
            num_negative += 1
        else:
            num_neutral += 1

    return num_positive, num_negative, num_neutral

# Initialize overall counters
overall_positive = 0
overall_negative = 0
overall_neutral = 0

Comment = df["Comment"]

for Comment in Comment:
    num_positive, num_negative, num_neutral = analyze_sentiment(Comment)
    overall_positive += num_positive
    overall_negative += num_negative
    overall_neutral += num_neutral


# Print overall sentiment counts
print(f"Overall Positive: {overall_positive}, Overall Negative: {overall_negative}, Overall Neutral: {overall_neutral}")

import matplotlib.pyplot as plt
import seaborn as sns

# Aggregate results
sentiments = {
    'Positive': overall_positive,
    'Negative': overall_negative,
    'Neutral': overall_neutral
}

# Plotting the results
sns.set(style="whitegrid")

# Create a bar plot
plt.figure(figsize=(6, 4))
sns.barplot(x=list(sentiments.keys()), y=list(sentiments.values()), palette="viridis")
plt.title("Sentiment Analysis Results")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Create a pie chart
plt.figure(figsize=(6, 4))
plt.pie(sentiments.values(), labels=sentiments.keys(), autopct='%1.1f%%', colors=sns.color_palette("viridis"))
plt.title("Sentiment Distribution")
plt.show()

"""#Further processing trying to interpret emojis
Work in Progress

"""
