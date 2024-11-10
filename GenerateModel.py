import numpy as np
import pandas as pd
import nltk

# For EDA and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# For text pre-processing
import re, string
import contractions
from unidecode import unidecode
from nltk.corpus import stopwords

nltk.download('stopwords')
import pickle
import os

# For tokenization and lemmatization
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')

# For model building
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load stopwords and customize
stop_words = stopwords.words('english')
stop_words.remove('not')

# Load dataset
df = pd.read_csv('Resources/Dataset/IMDB_Dataset.csv')
df.rename(columns={'review': 'text'}, inplace=True)


# Function to clean text data
def clean_noise_data(text, keepPunctuationMark=False):
    text = unidecode(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r"\b\d+\b", "", text)
    expanded_words = [contractions.fix(word) for word in text.split()]
    text = ' '.join(expanded_words)

    if not keepPunctuationMark:
        text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    else:
        punctuation_except_marks = string.punctuation.replace('?', '').replace('!', '')
        text = re.sub(r'[%s]' % re.escape(punctuation_except_marks), ' ', text)
        text = re.sub(r'(?<!\s)([?!])', r' \1', text)

    text = re.sub(r'\n', ' ', text)
    emoji_pattern = re.compile(r"["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Clean dataset and remove stopwords
data_cleaned_punctuation = df['text'].apply(lambda x: clean_noise_data(x, keepPunctuationMark=True))
cleaned_df_punctuation = data_cleaned_punctuation.to_frame(name='text')
cleaned_df_punctuation['sentiment'] = df['sentiment']
cleaned_df_punctuation['no_sw'] = cleaned_df_punctuation['text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

# Remove high-frequency neutral words
FREQWORDS = {'one', 'film', 'see', 'movie', 'even', 'time', 'story', 'character', 'make', 'scene', 'show', 'think', 'way'}


def remove_freqwords(text):
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])


cleaned_df_punctuation['no_sw_mostfreq_punctuation'] = cleaned_df_punctuation['no_sw'].apply(remove_freqwords)

# Lemmatize the data
lemmatizer = WordNetLemmatizer()
cleaned_df_punctuation['lemmatized'] = cleaned_df_punctuation['no_sw_mostfreq_punctuation'].apply(
    lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
)

# Prepare data for Naive Bayes
nb_df_punctuation = cleaned_df_punctuation.drop(columns=['text', 'no_sw', 'no_sw_mostfreq_punctuation'])
nb_df_punctuation.rename(columns={'lemmatized': 'review'}, inplace=True)
nb_df_punctuation['sentiment'] = [0 if each == 'negative' else 1 for each in nb_df_punctuation['sentiment']]

# Create feature matrix and labels
vectorizer = CountVectorizer(ngram_range=(2, 2))  # Use unigrams
X = vectorizer.fit_transform(nb_df_punctuation['review'])
y = nb_df_punctuation['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)

# Train Naive Bayes model
MNB = MultinomialNB()
model = MNB.fit(X_train, y_train)

# Check model accuracy
predicted = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predicted))

# Extract top features for positive and negative classes
feature_names = vectorizer.get_feature_names_out()
class_labels = model.classes_

# Extract the top positive and negative words and their log-probabilities
top_positive_words = sorted(zip(model.feature_log_prob_[1], feature_names), reverse=True)[:20]
top_negative_words = sorted(zip(model.feature_log_prob_[0], feature_names), reverse=True)[:20]

# Separate words and scores for positive and negative words
positive_scores, positive_words = zip(*top_positive_words)
negative_scores, negative_words = zip(*top_negative_words)

# Plot the top positive words
plt.figure(figsize=(10, 6))
plt.barh(positive_words, positive_scores, color='green')
plt.xlabel('Log Probability')
plt.title('Top 20 Words for Positive Sentiment')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()

# Plot the top negative words
plt.figure(figsize=(10, 6))
plt.barh(negative_words, negative_scores, color='red')
plt.xlabel('Log Probability')
plt.title('Top 20 Words for Negative Sentiment')
plt.gca().invert_yaxis()
plt.show()

# Save model and vectorizer
model_directory = './Model'
os.makedirs(model_directory, exist_ok=True)
model_path = os.path.join(model_directory, 'model.pkl')
with open(model_path, 'wb') as model_file:
    pickle.dump({
        'model': model,
        'vectorizer': vectorizer
    }, model_file)

print("Model and vectorizer saved successfully.")
