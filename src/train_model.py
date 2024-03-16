import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

disaster_tweets = [
    "There's a massive fire in my neighborhood! #fire #help",
    "Flooding is severe in my town. Roads are impassable. #flood #disaster",
]

non_disaster_tweets = [
    "Happy birthday to my best friend!",
    "Just finished watching a great movie. #weekendvibes",
]

tweets = disaster_tweets + non_disaster_tweets
labels = [1] * len(disaster_tweets) + [0] * len(non_disaster_tweets)

try:
  models_folder = "models"
  if not os.path.exists(models_folder):
      os.makedirs(models_folder)

  vectorizer = TfidfVectorizer(max_features=1000)
  vectorizer.fit(tweets)

  model = LogisticRegression()
  model.fit(vectorizer.transform(tweets), labels)

  pickle.dump(model, open(os.path.join(models_folder, "disaster_model.pkl"), "wb"))
  pickle.dump(vectorizer, open(os.path.join(models_folder, "disaster_vectorizer.pkl"), "wb"))

  print("Model and vectorizer trained and saved successfully!")
except ValueError as e:
  print("Error:", e)
