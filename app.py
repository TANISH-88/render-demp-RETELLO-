from flask import Flask, render_template, request, jsonify
import joblib, json, re, os, time
import numpy as np
from dotenv import load_dotenv
import requests

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

# Load .env
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

app = Flask(__name__)
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
        print("Predict error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/suggest", methods=["POST"])
def suggest():
    try:
        data = request.json or {}
        price = float(data.get("price", 0) or 0)

        if not GROQ_KEY:
            return jsonify({"error": "Groq API key not found in environment."}), 500

        def call_groq(prompt_payload):
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json=prompt_payload,
                timeout=25
            )
            resp.raise_for_status()
            j = resp.json()
            if isinstance(j, dict) and "choices" in j and len(j["choices"]) > 0:
                c0 = j["choices"][0]
                if isinstance(c0, dict):
                    if "message" in c0 and isinstance(c0["message"], dict):
                        return c0["message"].get("content", "")
                    if "text" in c0:
                        return c0.get("text", "")
            return ""

        user_prompt = (
            f"Monthly rent budget: ₹{price:,.2f}.\n"
            "You must reply ONLY with a JSON array of property suggestions.\n"
            "Each string should be in the format 'Property Name — Area, City'.\n"
            "Example:\n"
            "[\"Lodha World Towers — Lower Parel, Mumbai\", \"DLF The Crest — Gurgaon\"]\n\n"
            "Do not include explanations, comments, or any text outside the JSON array."
        )

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a strict assistant that only outputs JSON arrays of property suggestions."},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 220,
            "temperature": 0.3
        }

        # Send request to Groq
        raw = call_groq(payload)
        raw_clean = clean_text(raw)
        print("Groq raw output:", raw_clean)  # ✅ Debugging

        parsed = parse_json_array(raw_clean)

        # If invalid JSON, force retry
        if not parsed:
            retry_prompt = (
                "You failed to follow the output rule. Reply ONLY with a JSON array like this:\n"
                "[\"Lodha World Towers — Lower Parel, Mumbai\", \"DLF The Crest — Gurgaon\"]"
            )
            payload_retry = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "Respond only with JSON array of property suggestions."},
                    {"role": "user", "content": retry_prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.2
            }
            time.sleep(0.3)
            raw2 = call_groq(payload_retry)
            raw2_clean = clean_text(raw2)
            print("Retry output:", raw2_clean)
            parsed = parse_json_array(raw2_clean)

        if parsed:
            return jsonify({"suggestion": parsed})
        else:
            return jsonify({"error": "Groq returned invalid response", "raw": raw_clean}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
