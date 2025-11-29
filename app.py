from flask import Flask, render_template, request, jsonify
import joblib, json, re, os, time
import numpy as np
from dotenv import load_dotenv
import requests
from flask_cors import CORS

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<.*?>", "", text)
    return text.strip()

def parse_json_array(text: str):
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return [x.strip() for x in parsed if x.strip()]
    except Exception:
        pass
    return None

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")

app = Flask(__name__)
CORS(app)

model = joblib.load("rent_pipe.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = [[
            float(data["bedrooms"]), float(data["bathrooms"]), float(data["lotarea"]),
            float(data["grade"]), float(data["condition"]), float(data["waterfront"]),
            float(data["views"])
        ]]
        rent_log = model.predict(features)[0]
        rent = float(np.exp(rent_log))
        return jsonify({"prediction": rent, "message": f"Predicted Rent: ₹{rent:,.2f}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/suggest", methods=["POST"])
def suggest():
    try:
        data = request.json or {}
        price = float(data.get("price", 0) or 0)

        if not GROQ_KEY:
            return jsonify({"suggestion": ["Groq API key not found. Check Render environment."]})

        def call_groq(payload):
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )
            if resp.status_code != 200:
                raise Exception(f"Groq API returned {resp.status_code}")
            j = resp.json()
            if "choices" in j and len(j["choices"]) > 0:
                msg = j["choices"][0].get("message", {}).get("content", "")
                return msg.strip()
            return ""

        user_prompt = (
            f"Monthly rent budget: ₹{price:,.2f}.\n\n"
            "Find 3–5 realistic and unique luxury rental properties within this budget.\n"
            "Search globally — India, Dubai, Singapore, London, New York, etc.\n"
            "Each suggestion must include Property Name — Area, City, Country.\n"
            "Avoid repeating known names.\n"
            "Output only a JSON array of strings."
        )

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "Return only JSON arrays of property names."},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 280,
            "temperature": 0.8
        }

        raw = call_groq(payload)
        raw_clean = clean_text(raw)
        parsed = parse_json_array(raw_clean)

        if not parsed:
            retry_prompt = "Return 3–5 global properties strictly as a JSON array of strings."
            payload_retry = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "Return only valid JSON arrays."},
                    {"role": "user", "content": retry_prompt}
                ],
                "max_tokens": 250,
                "temperature": 0.85
            }
            time.sleep(0.5)
            raw2 = call_groq(payload_retry)
            raw2_clean = clean_text(raw2)
            parsed = parse_json_array(raw2_clean)

        if parsed:
            return jsonify({"suggestion": parsed})
        else:
            return jsonify({"suggestion": ["Unable to fetch live property suggestions."]})

    except Exception as e:
        return jsonify({"suggestion": [f"Error: {str(e)}"]})

if __name__ == "__main__":
    app.run(debug=True)

