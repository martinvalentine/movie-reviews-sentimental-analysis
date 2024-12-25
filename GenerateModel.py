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

from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

stop_words = stopwords.words('english')

dataframe = pd.read_csv('C:\\Users\\Martin\\Desktop\\movie-reviews-sentimental-analysis\\Resources\\Dataset\\IMDB_Dataset.csv')
dataframe.head()
dataframe.rename(columns={'review':'text'}, inplace = True)

def clean_noise_data(text):
    text = unidecode(text)

    text = text.lower()

    text = text.replace("!|s", "'s").replace("!ss", "'")

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>+', '', text)

    text = re.sub(r"\b\d+\b", "", text)

    expanded_words = [contractions.fix(word) for word in text.split()]
    text = ' '.join(expanded_words)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)

    text = re.sub(r'\n', ' ', text)

    text = re.sub(r'[’“”…]', '', text)

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

cleaned_data = dataframe['text'].apply(lambda x: clean_noise_data(x))
cleaned_df = cleaned_data.to_frame(name='text')
cleaned_df['sentiment'] = dataframe['sentiment']
cleaned_df['no_sw'] = cleaned_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

FREQWORDS = {'movie', 'film', 'one', 'would', 'time', 'even', 'story', 'see', 'could', 'get', 'people', 'made', 'make', 'way', 'movies', 'characters', 'character', 'films', 'two', 'think', 'watch', 'also', 'show', 'scene', 'look'}

def remove_freqwords(text):
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])


cleaned_df['no_sw_mostfreq'] = cleaned_df['no_sw'].apply(remove_freqwords)

lemmatizer = WordNetLemmatizer()
cleaned_df['lemmatized'] = cleaned_df['no_sw_mostfreq'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

final_df = cleaned_df.drop(columns=['text', 'no_sw', 'no_sw_mostfreq'])

final_df.rename(columns={'lemmatized': 'review'}, inplace=True)
final_df.sentiment = [0 if each == 'negative' else 1 for each in final_df.sentiment]

tokenized_reviews = final_df['review'].apply(lambda x : x.split())

vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(final_df['review'])
y = final_df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)

MNB = MultinomialNB()
model = MNB.fit(X_train, y_train)

predicted = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predicted))

feature_names = vectorizer.get_feature_names_out()
class_labels = model.classes_

top_positive_words = sorted(zip(model.feature_log_prob_[1], feature_names), reverse=True)[:20]
top_negative_words = sorted(zip(model.feature_log_prob_[0], feature_names), reverse=True)[:20]

positive_scores, positive_words = zip(*top_positive_words)
negative_scores, negative_words = zip(*top_negative_words)

plt.figure(figsize=(10, 6))
plt.barh(positive_words, positive_scores, color='green')
plt.xlabel('Log Probability')
plt.title('Top 20 Words for Positive Sentiment')
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(10, 6))
plt.barh(negative_words, negative_scores, color='red')
plt.xlabel('Log Probability')
plt.title('Top 20 Words for Negative Sentiment')
plt.gca().invert_yaxis()
plt.show()

model_directory = './Model'
os.makedirs(model_directory, exist_ok=True)
model_path = os.path.join(model_directory, 'model_final.pkl')
with open(model_path, 'wb') as model_file:
    pickle.dump({
        'model': model,
        'vectorizer': vectorizer
    }, model_file)

print("Model and vectorizer saved successfully.")
