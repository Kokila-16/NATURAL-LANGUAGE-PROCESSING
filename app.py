import os
import re
import pickle
import pandas as pd
import numpy as np
from flask import Flask, redirect, render_template_string, request
from textblob import TextBlob

# Initialize Flask App and Folders
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load your trained machine learning model and scaler for flirting detection
model = pickle.load(open("finalized_model_NLP.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

# Updated comprehensive compliment keywords
compliment_keywords = [
    "amazing", "brilliant", "wonderful", "fantastic", "incredible",
    "outstanding", "superb", "gorgeous", "handsome", "beautiful",
    "smart", "clever", "talented", "stunning", "charming",
    "kind", "sweet", "thoughtful", "appreciate", "grateful",
    "proud", "perfect", "legend", "rockstar", "best", "favorite", "special"
]

# Updated comprehensive flirting keywords
flirting_keywords = [
    "love", "cute", "handsome", "beautiful", "pretty", "attractive",
    "hot", "sexy", "miss you", "missing you", "date", "romantic",
    "romance", "jaan", "babu", "darling", "honey", "baby", "mine",
    "forever", "hug", "kiss", "wink", "blush", "crazy about you",
    "thinking of you", "my heart"
]

def analyze_chat(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    parsed_data = []
    for line in lines:
        name = None
        message = None

        # Pattern 1: iOS format -> [9/5/26, 9:32:14 AM] S V K: Hello
        match_bracket = re.match(r'^\[.*?\]\s*(.*?):\s*(.*)$', line)
        if match_bracket:
            name = match_bracket.group(1).strip()
            message = match_bracket.group(2).strip()
        else:
            # Pattern 2: Android format -> 9/5/26, 9:32:14 AM - S V K: Hello
            match_dash = re.match(r'^\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}.*?\s*-\s*(.*?):\s*(.*)$', line)
            if match_dash:
                name = match_dash.group(1).strip()
                message = match_dash.group(2).strip()

        if name and message:
            parsed_data.append({"Name": name, "Chat": message})
        elif parsed_data:
            parsed_data[-1]["Chat"] += " " + line

    df = pd.DataFrame(parsed_data)

    if df.empty or len(df) == 0:
        return 0, {"Neutral": 0, "Positive": 0, "Negative": 0}, [], {}, {}

    # Filter out system logs and calls
    df = df[
        ~df["Chat"].str.contains("Messages and calls are end-to-end encrypted|joined|left|security code|<Media omitted>", na=False, case=False) &
        ~df["Name"].str.contains("System|Call|Missed|Audio|Video", na=False, case=False)
    ]

    if len(df) == 0:
        return 0, {"Neutral": 0, "Positive": 0, "Negative": 0}, [], {}, {}

    # 1. Sentiment Analysis
    df["Sentiment_Score"] = df["Chat"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["Sentiment_Category"] = df["Sentiment_Score"].apply(
        lambda score: "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")
    )

    sentiment_counts = {"Neutral": 0, "Positive": 0, "Negative": 0}
    val_counts = df["Sentiment_Category"].value_counts().to_dict()
    sentiment_counts.update(val_counts)

    # 2. Compliments Keyword Matching
    def find_compliment_keyword(text):
        text_lower = str(text).lower()
        for word in compliment_keywords:
            if word in text_lower:
                return word
        return None

    df["Matched_Compliment"] = df["Chat"].apply(find_compliment_keyword)

    # 3. Flirting Detection using ML Model + Keyword Matching
    flirting_matches = []
    for chat_msg in df["Chat"]:
        text_lower = str(chat_msg).lower()
        matched_kw = None
        for word in flirting_keywords:
            if word in text_lower:
                matched_kw = word
                break

        if matched_kw:
            flirting_matches.append(matched_kw)
        else:
            # Fall back to ML model if no keyword found
            blob = TextBlob(str(chat_msg))
            compound = blob.sentiment.polarity
            neg = 0.0 if compound >= 0 else abs(compound)
            pos = compound if compound > 0 else 0.0
            neu = 1.0 - (neg + pos)
            topic = len(str(chat_msg).split())

            features = np.array([compound, neg, pos, neu, topic]).reshape(1, -1)
            scaled_features = scaler.transform(features)
            pred = model.predict(scaled_features)

            if pred[0] == 1:
                flirting_matches.append("ML Pattern")
            else:
                flirting_matches.append(None)

    df["Matched_Flirting"] = flirting_matches

    # Aggregate structured dictionaries
    total_messages = len(df)
    participants = df["Name"].unique().tolist()

    compliment_breakdown = {}
    flirting_breakdown = {}

    for name in participants:
        user_df = df[df["Name"] == name]

        comps = user_df[user_df["Matched_Compliment"].notnull()]["Matched_Compliment"].tolist()
        if comps:
            compliment_breakdown[name] = {"count": len(comps), "keywords": comps}

        flirts = user_df[user_df["Matched_Flirting"].notnull()]["Matched_Flirting"].tolist()
        if flirts:
            flirting_breakdown[name] = {"count": len(flirts), "keywords": flirts}

    return total_messages, sentiment_counts, participants, compliment_breakdown, flirting_breakdown

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            total_messages, sentiment_counts, participants, compliment_breakdown, flirting_breakdown = analyze_chat(file_path)

            html_template = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>WhatsApp Chat Analytics Dashboard</title>
                <script src="https://cdn.tailwindcss.com"></script>
            </head>
            <body class="bg-slate-50 text-slate-800 font-sans antialiased">
                <div class="max-w-4xl mx-auto px-4 py-10">
                    <div class="mb-8 text-center">
                        <h1 class="text-3xl font-extrabold text-slate-900 tracking-tight">WhatsApp Chat Analytics</h1>
                        <p class="text-sm text-slate-500 mt-1">Detailed sentiment, compliments, and flirting breakdown</p>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                        <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <h3 class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Total Messages</h3>
                            <p class="text-3xl font-bold text-indigo-600">{{ total_messages }}</p>
                        </div>
                        <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <h3 class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">Participants</h3>
                            <p class="text-lg font-medium text-slate-700">{% if participants %}{{ participants | join(', ') }}{% else %}None found{% endif %}</p>
                        </div>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                        <!-- Sentiment Card -->
                        <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <h3 class="text-lg font-semibold text-slate-900 mb-4 pb-2 border-b border-slate-100">Sentiment</h3>
                            <ul class="space-y-3">
                                {% for sentiment, count in sentiment_counts.items() %}
                                <li class="flex justify-between items-center text-sm">
                                    <span class="font-medium text-slate-600">{{ sentiment }}</span>
                                    <span class="bg-slate-100 text-slate-700 px-2.5 py-1 rounded-full font-semibold">{{ count }}</span>
                                </li>
                                {% endfor %}
                            </ul>
                        </div>

                        <!-- Compliments Card with Keywords -->
                        <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <h3 class="text-lg font-semibold text-slate-900 mb-4 pb-2 border-b border-slate-100">Compliments</h3>
                            {% if compliment_breakdown %}
                            <div class="space-y-4">
                                {% for name, data in compliment_breakdown.items() %}
                                <div>
                                    <div class="flex justify-between items-center text-sm mb-1">
                                        <span class="font-bold text-slate-700">{{ name }}</span>
                                        <span class="bg-emerald-50 text-emerald-600 px-2.5 py-0.5 rounded-full font-semibold text-xs">{{ data.count }}</span>
                                    </div>
                                    <div class="flex flex-wrap gap-1 mt-1">
                                        {% for kw in data.keywords %}
                                        <span class="bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs px-2 py-0.5 rounded-md font-medium">{{ kw }}</span>
                                        {% endfor %}
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                            {% else %}
                            <p class="text-sm text-slate-400 italic">No compliments detected.</p>
                            {% endif %}
                        </div>

                        <!-- Flirting Card with Keywords -->
                        <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <h3 class="text-lg font-semibold text-slate-900 mb-4 pb-2 border-b border-slate-100">Flirting Detection</h3>
                            {% if flirting_breakdown %}
                            <div class="space-y-4">
                                {% for name, data in flirting_breakdown.items() %}
                                <div>
                                    <div class="flex justify-between items-center text-sm mb-1">
                                        <span class="font-bold text-slate-700">{{ name }}</span>
                                        <span class="bg-pink-50 text-pink-600 px-2.5 py-0.5 rounded-full font-semibold text-xs">{{ data.count }}</span>
                                    </div>
                                    <div class="flex flex-wrap gap-1 mt-1">
                                        {% for kw in data.keywords %}
                                        <span class="bg-pink-50 text-pink-700 border border-pink-200 text-xs px-2 py-0.5 rounded-md font-medium">{{ kw }}</span>
                                        {% endfor %}
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                            {% else %}
                            <p class="text-sm text-slate-400 italic">No flirting detected.</p>
                            {% endif %}
                        </div>
                    </div>

                    <div class="text-center">
                        <a href="/" class="inline-block bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-6 py-2.5 rounded-xl transition shadow-sm">Analyze Another Chat</a>
                    </div>
                </div>
            </body>
            </html>
            """
            return render_template_string(
                html_template,
                total_messages=total_messages,
                participants=participants,
                sentiment_counts=sentiment_counts,
                compliment_breakdown=compliment_breakdown,
                flirting_breakdown=flirting_breakdown
            )

    upload_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload WhatsApp Chat</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-50 text-slate-800 font-sans antialiased flex items-center justify-center min-h-screen">
        <div class="bg-white p-8 rounded-2xl shadow-sm border border-slate-200 max-w-md w-full text-center">
            <h2 class="text-2xl font-bold text-slate-900 mb-2">WhatsApp Chat Analyzer</h2>
            <p class="text-sm text-slate-500 mb-6">Upload your exported chat `.txt` file to view deep insights.</p>

            <form method="POST" enctype="multipart/form-data" class="space-y-4">
                <div class="border-2 border-dashed border-slate-200 rounded-xl p-6 hover:border-indigo-500 transition cursor-pointer">
                    <input type="file" name="file" accept=".txt" class="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-600 hover:file:bg-indigo-100">
                </div>
                <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2.5 rounded-xl transition shadow-sm">Analyze Chat</button>
            </form>
        </div>
    </body>
    </html>
    """
    return render_template_string(upload_template)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
