import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Function to read a file and return content
def read_file(file_path):
    """
    Reads the content of a file given its file path.

        Parameters:
            file_path (str): The path to the file to be read.

        Returns:
            str: The content of the file as a string.

        Raises:
            FileNotFoundError: If the file does not exist, prints an 
                                error message and exits the program.
    """
    try:
        with open(file_path, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        exit()

# Function to preprocess text
def preprocess_text(text, language="english"):
    """
    Convert text to lowercase, remove punctuation, and tokenize it.
    
    Parameters:
    - text (str): String to preprocess.
    - language (str, optional): Language for tokenization. Defaults to
      "english".
    
    Returns:
    - list: List of tokens from the processed text.
    """
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text, language)

# Function to remove stopwords from a list of words
def filter_stopwords(words, language):
    """
    Filters out stop words from a list of words based on the specified
    language.

    Parameters:
    words (list of str): The list of words from which to filter out stop
    words.
    language (str): The language for which to filter stop words, e.g., 
    'english'.

    Returns:
    list of str: A list of words with stop words removed.
    """
    stop_words = set(stopwords.words(language))
    return [word for word in words if word not in stop_words]

# Function to read and parse emotions from a file
def parse_emotions(file_path, filtered_words):
    """
    Parses a file to extract emotions associated with specific
    words. Filters out words not in the provided list.

    Parameters:
    - file_path (str): The path to the file containing word-emotion
                       pairs on separate lines, formatted as
                       "word:emotion".
    - filtered_words (list): List of words to filter the emotions
                             by.

    Returns:
    - emotion_list (list): A list of emotions corresponding to the
                           filtered words found in the file.
    """
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
    """
    Analyzes the sentiment of a given text.

    This function takes a string input, calculates the sentiment
    score using SentimentIntensityAnalyzer, and prints whether
    the sentiment is positive, negative, or neutral based on the
    computed scores.

    Parameters:
    text (str): The text to analyze sentiment for.

    Returns:
    None
    """
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    if score['neg'] > score['pos']:
        print("Negative Sentiment")
    elif score['pos'] > score['neg']:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")

# Function to get the sentiment of a word
def get_word_sentiment(word):
    """
    Analyzes the sentiment of a given word using sentiment intensity.
    Parameters:
        word (str): The word to analyze the sentiment of.
    Returns:
        str: 'positive', 'negative', or 'neutral' based on the sentiment
             score of the word.
    """
    sentiment_scores = SentimentIntensityAnalyzer().polarity_scores(word)
    if sentiment_scores['compound'] > 0.1:
        return 'positive'
    elif sentiment_scores['compound'] < -0.1:
        return 'negative'
    else:
        return 'neutral'

# Function to map emotions to colors
def map_emotions_to_colors(emotion_counts):
    """
    Maps emotional sentiments to corresponding colors.

    Args:
        emotion_counts: A dictionary with emotions as keys and their counts as values.

    Returns:
        A list of strings representing colors associated with each emotional sentiment.
    """
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