from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🔤 Cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    cleaned = clean_text(message)

    data = vectorizer.transform([cleaned])
    prediction = model.predict(data)[0]

    # Confidence
    try:
        prob = model.predict_proba(data)[0]
        confidence = max(prob) * 100
    except:
        confidence = 0

    result = "Spam ❌" if prediction == "spam" else "Not Spam ✅"

    return render_template(
        "index.html",
        prediction_text=result,
        confidence=f"{confidence:.2f}%"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)