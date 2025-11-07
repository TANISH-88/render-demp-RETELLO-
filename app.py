from flask import Flask, render_template, request, jsonify
import joblib, json, re, os, time
import numpy as np
from dotenv import load_dotenv
import requests

# ------------------ Helper Functions ------------------

def clean_text(text: str) -> str:
    """Cleans AI output of unwanted tags or junk."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<.*?>", "", text)
    return text.strip()

def parse_json_array(text: str):
    """Parses only valid JSON arrays of strings."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return [x.strip() for x in parsed if x.strip()]
    except Exception:
        pass
    return None

# ------------------ Environment Setup ------------------

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")

# ✅ Correct new endpoint for Groq
GROQ_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

# ✅ Use the correct updated model name
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")

# ✅ Debug log to confirm key loaded
print("Loaded GROQ_API_KEY:", bool(GROQ_KEY))
print("Using Groq URL:", GROQ_URL)
print("Model in use:", GROQ_MODEL)

app = Flask(__name__)
model = joblib.load("rent_pipe.pkl")

# ------------------ Routes ------------------

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

# ------------------ GROQ Suggestion Logic ------------------

@app.route("/suggest", methods=["POST"])
def suggest():
    try:
        data = request.json or {}
        price = float(data.get("price", 0) or 0)

        if not GROQ_KEY:
            return jsonify({"suggestion": ["Groq API key not found. Check Render environment."]})

        def call_groq(prompt_payload):
            """Send request to Groq API."""
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_KEY}",
                    "Content-Type": "application/json"
                },
                json=prompt_payload,
                timeout=30
            )
            if resp.status_code != 200:
                print("Groq API Error:", resp.status_code, resp.text)
                raise Exception(f"Groq API returned {resp.status_code}")
            
            j = resp.json()
            print("🔍 GROQ RAW:", j)  # Debug: show Groq response in logs

            if "choices" in j and len(j["choices"]) > 0:
                msg = j["choices"][0].get("message", {}).get("content", "")
                return msg.strip()
            return ""

        # ------------------ AI Prompt ------------------
        user_prompt = (
            f"Monthly rent budget: ₹{price:,.2f}.\n\n"
            "Find 3–5 **realistic and unique luxury rental properties** within this budget.\n"
            "Search globally — India, Dubai, Singapore, London, New York, etc.\n"
            "Each suggestion must include **Property Name — Area, City, Country**.\n"
            "Avoid repeating any known names like Lodha, DLF, Prestige, etc.\n"
            "Output only a JSON array of strings. Example:\n"
            "[\"The Address Residence — Downtown, Dubai, UAE\", \"Vasant Vihar Villas — New Delhi, India\"]"
        )

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a global real estate AI assistant that outputs only JSON arrays of property names."
                },
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 280,
            "temperature": 0.8
        }

        # First attempt
        raw = call_groq(payload)
        raw_clean = clean_text(raw)
        parsed = parse_json_array(raw_clean)

        # Retry if needed
        if not parsed:
            retry_prompt = (
                "Retry and return 3–5 unique global properties as a valid JSON array of strings. "
                "Do not include explanations or text."
            )
            payload_retry = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "Return only valid JSON arrays of global property names."},
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
            return jsonify({"suggestion": ["Unable to fetch live property suggestions. Please retry."]})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"suggestion": [f"Error: {str(e)}"]})

# ------------------ Run ------------------

if __name__ == "__main__":
    app.run(debug=True)
