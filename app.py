from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("books_model.pkl")

# MUST match training-time feature order exactly
MODEL_COLUMNS = [
    "book_edition",

    "Accounting Information Systems",
    "Commercial Law - Fresh Perspectives",
    "Economics Volume 1: Global and Southern African Perspectives",
    "Financial Accounting: An introduction",
    "Financial Accounting: The Question Book",
    "Managerial Finance",

    "Acceptable",
    "Average",
    "Excellent"
]


def prepare_input(book_title, book_edition, quality):
    # initialise all features to 0
    data = dict.fromkeys(MODEL_COLUMNS, 0)

    # numeric feature
    data["book_edition"] = int(book_edition)

    # one-hot encode title
    if book_title in data:
        data[book_title] = 1

    # one-hot encode quality
    if quality in data:
        data[quality] = 1

    return pd.DataFrame([data])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    book_title = request.form["book_title"]
    book_edition = request.form["book_edition"]
    quality = request.form["quality"]
    listed_price = float(request.form["listed_price"])

    X = prepare_input(book_title, book_edition, quality)

    predicted_price = model.predict(X)[0]

    # Deal evaluation logic
    if listed_price < predicted_price * 0.9:
        verdict = "Good deal"
        pricing = "underpriced"
    elif listed_price > predicted_price * 1.1:
        verdict = "Bad deal"
        pricing = "overpriced"
    else:
        verdict = "Fair deal"
        pricing = "fairly priced"

    result = (
        f"This is a {verdict} as the book is {pricing} "
        f"at R{listed_price:.0f}. Estimated fair value is R{predicted_price:.0f}."
    )

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)