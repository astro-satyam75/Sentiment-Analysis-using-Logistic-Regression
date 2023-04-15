import pandas as pd
import numpy as np
import nltk
import io
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from afinn import Afinn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, jsonify, Response

# Downloading stopwords
nltk.download('stopwords')

# Operations on the datasets
df1 = pd.read_csv(r'D:\SATYAM\IMDB_analysis\rotten_tomatoes_critic_reviews.csv')
df2 = pd.read_csv(r'D:\SATYAM\IMDB_analysis\rotten_tomatoes_movies.csv')
df_r = pd.merge(df1, df2, on='rotten_tomatoes_link')
df3 = df_r[['movie_title', 'review_content']]
df3.dropna(inplace=True)
df3['review_content'] = df3['review_content'].apply(lambda x: str(x))
df3['review_content'] = df3['review_content'].apply(lambda x: f'"{x}"' if not x.startswith('"') else x)
df = df3.groupby('movie_title')['review_content'].apply(list).reset_index()

# Data Preprocessing
df['review_content'] = df['review_content'].apply(lambda x: ''.join(x))  # join list of words into a string
df['review_content'] = df['review_content'].apply(lambda x: x.lower())  # convert to lowercase
df['review_content'] = df['review_content'].apply(lambda x: word_tokenize(x))  # tokenize into a list of words
stop_words = set(stopwords.words('english'))
df['review_content'] = df['review_content'].apply(
    lambda x: [word for word in x if not word in stop_words])  # remove stopwords
df['review_content'] = df['review_content'].apply(lambda x: ' '.join(x))  # join back into a string

# Labeling data using AFINN sentiment lexicon
afinn = Afinn()
labels = np.where(df['review_content'].apply(afinn.score) >= 0, 1, 0)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review_content'], labels, test_size=0.2, random_state=42)

# Feature Extraction
tfidf_vectorizer = TfidfVectorizer()
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Building the Model
model = LogisticRegression()
model.fit(tfidf_train, y_train)

# Evaluating the Model
y_pred = model.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1)

# Flask App
app = Flask(__name__, static_folder='static')


@app.route('/')
def index():
    movies = df['movie_title'].tolist()
    return render_template('index.html', movies=movies)


@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    # Get selected movie from form
    movie = request.form['movie']
    reviews = df.loc[df['movie_title'] == movie, 'review_content'].tolist()

    # Predict sentiment for each review
    sentiments = []
    for review in reviews:
        review = review.lower()
        review = word_tokenize(review)
        review = [word for word in review if not word in stop_words]
        review = ' '.join(review)
        if review:
            review_tfidf = tfidf_vectorizer.transform([review])
            prediction = model.predict(review_tfidf)
            if prediction[0] == 1:
                sentiment = 'Positive'
            else:
                sentiment = 'Negative'
            sentiments.append(sentiment)

    # Get the list of movies
    movies = df['movie_title'].unique().tolist()

    # Render the template with the results
    return render_template('index.html', movie=movie, movies=movies, reviews=reviews, sentiments=sentiments,
                           accuracy=accuracy, precision=precision, recall=recall, f1_score=f1)


if __name__ == '__main__':
    app.run(debug=True)
