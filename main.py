import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Function to read a file and return content
def read_file(file_path):
    try:
        with open(file_path, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        exit()

# Function to preprocess text
def preprocess_text(text, language="english"):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text, language)

# Function to remove stopwords from a list of words
def filter_stopwords(words, language):
    stop_words = set(stopwords.words(language))
    return [word for word in words if word not in stop_words]

# Function to read and parse emotions from a file
def parse_emotions(file_path, filtered_words):
    emotion_list = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                word, emotion = clear_line.split(':')
                if word in filtered_words:
                    emotion_list.append(emotion)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        exit()
    return emotion_list

# Function to perform sentiment analysis
def sentiment_analyse(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    if score['neg'] > score['pos']:
        print("Negative Sentiment")
    elif score['pos'] > score['neg']:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")

# Function to get the sentiment of a word
def get_word_sentiment(word):
    sentiment_scores = SentimentIntensityAnalyzer().polarity_scores(word)
    if sentiment_scores['compound'] > 0.1:
        return 'positive'
    elif sentiment_scores['compound'] < -0.1:
        return 'negative'
    else:
        return 'neutral'

# Function to map emotions to colors
def map_emotions_to_colors(emotion_counts):
    color_map = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
    return [color_map[get_word_sentiment(emotion)] for emotion in emotion_counts.keys()]

# Main analysis
text = read_file("read.txt")

tokenized_words = preprocess_text(text, "english")

filtered_words = filter_stopwords(tokenized_words, 'english')
emotion_list = parse_emotions("emotions.txt", filtered_words)
emotion_counts = Counter(emotion_list)
mapped_colors = map_emotions_to_colors(emotion_counts)

# Plotting
fig, ax = plt.subplots()
ax.bar(emotion_counts.keys(), emotion_counts.values(), color=mapped_colors)
fig.autofmt_xdate()

if emotion_counts:
    average = sum(emotion_counts.values()) / len(emotion_counts)
    ax.axhline(average, color='blue', linewidth=2, label=f'Average: {average:.2f}')
else:
    print("No emotions were found to calculate the average.")

ax.legend()
#plt.savefig('graph.png')
plt.show()

sentiment_analyse(text)