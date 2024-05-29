## YouTube Comments Sentiment Analysis

This project involves fetching comments from a YouTube video and performing sentiment analysis on them. The workflow is as follows:

1. **Fetch Comments from YouTube Video**: Using the YouTube Data API, comments from a specified video are fetched and saved to a CSV file.
2. **Sentiment Analysis**: Utilizing the NLTK library's VADER sentiment analyzer, each comment is analyzed to determine its sentiment (positive, negative, or neutral).
3. **Visualization**: The results of the sentiment analysis are visualized using bar plots and pie charts to display the distribution of sentiments.

### Steps and Code Explanation

1. **Setup YouTube Data API**:
   - Create a YouTube Data API client with your API key.
   - Define the video ID from which you want to fetch comments.
   
2. **Fetch Comments and Save to CSV**:
   - Use the YouTube Data API to fetch comments.
   - Save the fetched comments to a CSV file (`youtube_comments.csv`).

```python
import requests
import csv
import pandas as pd
import os
from googleapiclient.discovery import build

# Set up the YouTube Data API key
api_key = "YOUR_API_KEY"  # Replace with your own API key

# Create a YouTube Data API client
youtube = build("youtube", "v3", developerKey=api_key)

# Define the video ID for the video you want to fetch comments from
video_id = "VIDEO_ID"

def fetch_comments_to_csv(api_key, video_id, max_results=1000):
    csv_filename = "youtube_comments.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Author", "Comment", "Likes"])
        page_token = None
        total_comments = 0

        while total_comments < max_results:
            comments = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - total_comments),
                textFormat="plainText",
                pageToken=page_token
            ).execute()

            for comment in comments.get("items", []):
                snippet = comment["snippet"]["topLevelComment"]["snippet"]
                author_name = snippet["authorDisplayName"]
                comment_text = snippet["textDisplay"]
                like_count = snippet["likeCount"]
                csv_writer.writerow([author_name, comment_text, like_count])
                total_comments += 1

            if "nextPageToken" in comments:
                page_token = comments["nextPageToken"]
            else:
                break

fetch_comments_to_csv(api_key, video_id, max_results=1000)
```

3. **Sentiment Analysis**:
   - Read the saved CSV file.
   - Use the NLTK library to perform sentiment analysis on each comment.
   - Count the number of positive, negative, and neutral comments.

```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import pandas as pd

# Read the CSV file
df = pd.read_csv('youtube_comments.csv')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(Comment):
    sentences = tokenize.sent_tokenize(Comment)
    num_positive = 0
    num_negative = 0
    num_neutral = 0

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

overall_positive = 0
overall_negative = 0
overall_neutral = 0

for Comment in df["Comment"]:
    num_positive, num_negative, num_neutral = analyze_sentiment(Comment)
    overall_positive += num_positive
    overall_negative += num_negative
    overall_neutral += num_neutral

print(f"Overall Positive: {overall_positive}, Overall Negative: {overall_negative}, Overall Neutral: {overall_neutral}")
```

4. **Visualization**:
   - Visualize the sentiment analysis results using bar plots and pie charts.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sentiments = {
    'Positive': overall_positive,
    'Negative': overall_negative,
    'Neutral': overall_neutral
}

sns.set(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.barplot(x=list(sentiments.keys()), y=list(sentiments.values()), palette="viridis")
plt.title("Sentiment Analysis Results")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
plt.pie(sentiments.values(), labels=sentiments.keys(), autopct='%1.1f%%', colors=sns.color_palette("viridis"))
plt.title("Sentiment Distribution")
plt.show()
```

### Results

The sentiment analysis of the comments resulted in:
- **Positive Comments**: 309
- **Negative Comments**: 58
- **Neutral Comments**: 867

The visualizations provide a clear picture of the sentiment distribution among the comments.

### Requirements

- Python 3.x
- pandas
- google-api-python-client
- nltk
- matplotlib
- seaborn

### How to Run

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Replace the `api_key` and `video_id` with your own values in the code.
4. Run the script to fetch comments and perform sentiment analysis.

### Conclusion

This project demonstrates how to use the YouTube Data API to fetch comments and perform sentiment analysis using NLTK's VADER sentiment analyzer, providing insights into the viewers' sentiments.
