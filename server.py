from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS so your website can call this server

# Initialize Hugging Face client
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token="your_huggingface_token_here"  # 🔐 Replace this with your actual token
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the following information when answering: Frank is a 38-year-old data analyst turned independent game developer from Portland, Oregon. With a degree in Applied Mathematics and a minor in Philosophy, he has a knack for blending logic with storytelling. His career began in corporate analytics, but his passion for creativity led him to pivot into the indie game scene."},
        {"role": "user", "content": user_message}
    ]

    response = client.chat_completion(messages, max_tokens=200)
    return jsonify({"reply": response.choices[0].message["content"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Port 10000 is recommended for Render
