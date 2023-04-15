## ROTTEN TOMATOES MOVIE REVIEW SENTIMENT ANALYZER USING LOGISTIC REGRESSION

This is a Flask web application that performs sentiment analysis on movie reviews using the Logistic Regression algorithm.

The code first reads in two CSV files, rotten_tomatoes_critic_reviews.csv and rotten_tomatoes_movies.csv, merges them on the rotten_tomatoes_link column, and extracts the movie_title and review_content columns. The reviews are grouped by movie title and processed for text preprocessing, which includes converting to lowercase, tokenizing into words, removing stop words, and converting back into a string.

The AFINN sentiment lexicon is used to label the data as either positive or negative based on the sentiment score of the review. The data is then split into training and testing sets, and a TfidfVectorizer is used to extract features from the text data.

A logistic regression model is built using the training data and evaluated using the testing data. The accuracy, precision, recall, and F1-score of the model are printed to the console.

The Flask app renders an HTML template that displays a dropdown list of movie titles. When the user selects a movie and submits the form, the app retrieves the reviews for that movie and predicts the sentiment of each review using the trained model. The predicted sentiment is displayed for each review, along with the accuracy, precision, recall, and F1-score of the model.





