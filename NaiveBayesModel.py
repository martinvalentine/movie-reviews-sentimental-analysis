import numpy as np
import pandas as pd
import nltk # Import natural Language Toolkit lib

# For EDA
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# For text pre-processing
import re, string # Import re and string for regular expressions and string operations
import contractions # Import contractions for expanding contractions
from unidecode import unidecode  # Import unidecode for Unicode normalization
from nltk.corpus import stopwords
nltk.download('stopwords') # For stopwords
import pickle
import os

# For tokenization and lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') # For Lemmatization
nltk.download('punkt') # For tokenization

# For model building
from sklearn.feature_extraction.text import CountVectorizer # Feature extraction for Naive Bayes
from sklearn.model_selection import train_test_split # For splitting the data into training and testing sets
from sklearn.naive_bayes import BernoulliNB # Naive Bayes model
from sklearn.naive_bayes import MultinomialNB # Naive Bayes model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score # For model evaluation

stop_words = stopwords.words('english') # Get the stopwords
# Remove 'not' from the stopwords list
stop_words.remove('not')

df=pd.read_csv('Resources/Dataset/IMDB_Dataset.csv')
df.rename(columns={'review':'text'}, inplace = True)


# Remove special characters, numbers, and punctuation
def clean_noise_data(text, keepPunctuationMark=False):
    # Use unidecode to normalize special characters to ASCII equivalents such as é -> e, ñ -> n, etc.
    text = unidecode(text)

    # Handle word after use unicode to ASCII, like it s -> it's for expanding contractions
    # Todo: Add more contractions

    # Convert text to lowercase for uniformity
    text = text.lower()

    # Replace special characters (e.g., "!|s" with "'s")
    text = text.replace("!|s", "'s").replace("!ss", "'")

    # Remove URLs and HTML tags to clean the text
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)

    # Remove numbers to focus on textual data
    text = re.sub(r"\b\d+\b", "", text)

    # Expand contractions for consistent spacing
    expanded_words = [contractions.fix(word) for word in text.split()]
    text = ' '.join(expanded_words)

    # Handle punctuation based on keepPunctuationMark setting
    if not keepPunctuationMark:
        # Replace all punctuation with spaces
        text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    else:
        # Remove all punctuation except ? and !, replacing other punctuation with spaces
        punctuation_except_marks = string.punctuation.replace('?', '').replace('!', '')
        text = re.sub(r'[%s]' % re.escape(punctuation_except_marks), ' ', text)

        # Add a space before ? and ! if they're attached to a word
        text = re.sub(r'(?<!\s)([?!])', r' \1', text)

    # Remove newline characters for cleaner text
    text = re.sub(r'\n', ' ', text)

    # Remove specific special characters that are not needed (such as curly quotes)
    text = re.sub(r'[’“”…]', '', text)

    # Define a pattern to remove emojis from the text
    emoji_pattern = re.compile(r"["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    # Remove emojis from the text
    text = emoji_pattern.sub(r'', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()  # Also trim leading and trailing spaces

    return text

data_cleaned_punctuation = df['text'].apply(lambda x: clean_noise_data(x, keepPunctuationMark=True))
cleaned_df_punctuation = data_cleaned_punctuation.to_frame(name='text')
cleaned_df_punctuation['sentiment'] = df['sentiment']

cleaned_df_punctuation['no_sw'] = cleaned_df_punctuation['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Function to find the most frequent words in a specified column of a DataFrame
def find_most_frequent_words(df, column_name, top_n=10):
    counter = Counter()
    for text in df[column_name].values:
        for word in text.split():
            counter[word] += 1
    most_common_words = counter.most_common(top_n)
    return pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
most_common_with_punctuation = find_most_frequent_words(cleaned_df_punctuation, 'no_sw', 15)

FREQWORDS = {'story', 'movie', 'see', 'time', 'one', 'even', 'film'}

def remove_freqwords(text):
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

# Apply the function to remove the most common words:
cleaned_df_punctuation['no_sw_mostfreq_punctuation'] = cleaned_df_punctuation['no_sw'].apply(remove_freqwords)
lemmatizer = WordNetLemmatizer()
cleaned_df_punctuation['lemmatized'] = cleaned_df_punctuation['no_sw_mostfreq_punctuation'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
nb_df_punctuation = cleaned_df_punctuation.drop(columns = ['text', 'no_sw', 'no_sw_mostfreq_punctuation'])
nb_df_punctuation.rename(columns={'lemmatized': 'review'}, inplace=True)
nb_df_punctuation.sentiment = [0 if each == 'negative' else 1 for each in nb_df_punctuation.sentiment]
tokenized_reviews_punctuation = nb_df_punctuation['review'].apply(lambda x : x.split())

vectorizer = CountVectorizer(ngram_range=(2, 2))
word_count_matrix_bi_punctuation = vectorizer.fit_transform(nb_df_punctuation['review'])

y_punctuation = nb_df_punctuation['sentiment']
X_bi_punctuation = word_count_matrix_bi_punctuation
X_train_bi_punctuation, X_test_bi_punctuation, y_train_bi_punctuation, y_test_bi_punctuation = train_test_split(X_bi_punctuation, y_punctuation, test_size=0.20, random_state=30)

MNB = MultinomialNB()
model = MNB.fit(X_train_bi_punctuation, y_train_bi_punctuation)
predicted = model.predict(X_test_bi_punctuation)
print(accuracy_score(y_test_bi_punctuation, predicted))

# Check if the directory exists, if not, create it
model_directory = './Model'
os.makedirs(model_directory, exist_ok=True)

# Define the path where the model will be saved
model_filename = 'model.pkl'
model_path = os.path.join(model_directory, model_filename)
# Save the model and vectorizer together in a dictionary

with open(model_path, 'wb') as model_file:
    pickle.dump({
        'model': model,  # Trained model
        'vectorizer': vectorizer  # The vectorizer used to transform reviews
    }, model_file)

print("Model and vectorizer saved successfully.")



