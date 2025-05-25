from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

hf_token = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=hf_token
)

# A dictionary of predefined topic knowledge chunks
TOPIC_KNOWLEDGE = {
    "career": """
        Frank transitioned from a corporate data analyst to an indie game developer 
        because he wanted more creative freedom and to pursue storytelling through games.
    """,
    "education": """
        Frank holds a degree in Applied Mathematics and a minor in Philosophy. 
        This background helps him design logic-based gameplay with deep narratives.
    """,
    "games": """
        Frank develops puzzle-based and narrative-driven indie games. 
        His recent work includes a mystery puzzle game inspired by real-world events.
    """
}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").lower()

    # Detect topic based on keywords
    if any(keyword in user_message for keyword in ["career", "switch", "change", "job"]):
        topic = "career"
    elif any(keyword in user_message for keyword in ["study", "education", "degree", "background"]):
        topic = "education"
    elif any(keyword in user_message for keyword in ["game", "project", "build", "create"]):
        topic = "games"
    else:
        topic = None

    if topic:
        system_content = f"""
        You are a helpful assistant. Provide a short, clear answer using the following information:
        {TOPIC_KNOWLEDGE[topic]}
        Also, suggest one or two follow-up questions the user can ask next, as options.
        Format them like:
        - Option 1: [question text]
        - Option 2: [question text]
        """
    else:
        system_content = """
        You are a helpful assistant. Frank is a 38-year-old data analyst turned indie game developer.
        Ask the user to pick a topic they're interested in learning more about:
        - Career
        - Education
        - Games
        """

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message}
    ]

    response = client.chat_completion(messages, max_tokens=250)
    return jsonify({"reply": response.choices[0].message["content"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
