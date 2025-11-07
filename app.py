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
GROQ_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

# ✅ Debug line
print("Loaded GROQ_API_KEY:", bool(GROQ_KEY))

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
            return jsonify({"suggestion": ["Groq API key not found. Check environment settings on Render."]})

        def call_groq(prompt_payload):
            """Send request to Groq API."""
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json=prompt_payload,
                timeout=30
            )
            resp.raise_for_status()
            j = resp.json()
            if "choices" in j and len(j["choices"]) > 0:
                msg = j["choices"][0].get("message", {}).get("content", "")
                return msg.strip()
            return ""

        # ------------------ Dynamic Prompt ------------------
        user_prompt = (
            f"Monthly rent budget: ₹{price:,.2f}.\n\n"
            "Find 3–5 realistic **luxury or premium rental properties** within this budget.\n"
            "Search across India and globally (like Dubai, Singapore, London, New York, etc.).\n"
            "Each suggestion must include **Property Name — Area, City, Country**.\n"
            "Avoid repetition and avoid using known names like Lodha, DLF, Prestige, etc.\n"
            "Output strictly as a JSON array of strings. Example only:\n"
            "[\"The Address Residence — Downtown, Dubai, UAE\", \"Vasant Vihar Homes — New Delhi, India\"]\n"
            "No extra text or explanation — JSON array only."
        )

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a real estate AI that outputs ONLY JSON arrays of global properties."},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 280,
            "temperature": 0.8
        }

        # First attempt
        raw = call_groq(payload)
        raw_clean = clean_text(raw)
        parsed = parse_json_array(raw_clean)

        # Retry if invalid
        if not parsed:
            retry_prompt = (
                "Retry: Return 3–5 unique, real-sounding luxury properties worldwide. "
                "Output only JSON array of strings in the same format."
            )
            payload_retry = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "Return only JSON arrays with property suggestions."},
                    {"role": "user", "content": retry_prompt}
                ],
                "max_tokens": 240,
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
        return jsonify({"suggestion": [f"Error: {str(e)}"]})

# ------------------ Run ------------------

if __name__ == "__main__":
    app.run(debug=True)
