from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Replace with your trained model
import pickle

model = pickle.load(open("models/disaster_model.pkl", "rb"))
vectorizer = pickle.load(open("models/disaster_vectorizer.pkl", "rb"))

app = Flask(__name__, template_folder="templates")


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    try:
        vectorized_text = vectorizer.transform([text])
        prediction = model.predict(vectorized_text)[0]
        return render_template("result.html", prediction=prediction)
    except (ValueError, pickle.UnpicklingError) as e:
        error_message = "An error occurred. Please ensure you have trained and saved the model and vectorizer correctly."
        return render_template("error.html", error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)