from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
from flask_cors import CORS
import os
from openai import OpenAI

app = Flask(__name__)
CORS(app)

oai_token = os.getenv("OAI_TOKEN")

client = OpenAI(api_key=oai_token)

# A dictionary of predefined topic knowledge chunks
txt = "Joseph, a campus hire at Deloitte USI since Jan 2023, works in AI & Data Engineering. Skilled in Databricks, Azure, ETL/ML pipelines, Unity Catalog, and Feature Store. Enthusiastic learner with strong data intuition. Enjoys snooker, tennis, and trekking.Skills(Tech: Python, Pandas, ML, Databricks MLflow, SQL, PySpark, Azure, Power BI, Excel, R, C++Other: Agile, Communication, Teamwork, Planning) Experience(Built data quality framework in Databricks for US insurer: cut failures 60%, saved $2M/year.QA on Azure healthcare data layer; migrated Shell to PySpark for provider data ingestion.) Projects(Fruit decay detector using YOLOv4 (98% accuracy).Oxygen concentrator for COVID, deployed in hospital.Movie recommender (ALS, Databricks), Hackathon finalist.) Publication(YOLOv4 fruit decay paper – IEEE 2024 (pending).)Awards & Certifications(2× Deloitte Applause Award. Azure Certified Cloud Practitioner. Databricks ML Associate) Education (B.Tech EEE, GEC Thrissur (2018–2022), GPA 8.39. Led SAE E-BAJA, IEDC Hackathon winner, active NSS member.)"
system_content = f"""
        You are a helpful person named Joseph. Provide a short, clear answer using the following information:
        {txt}
        Give bullet points if required.
        Suggest one follow-up question the user can ask next. It should come in a new line.
        """
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").lower()

    
completion = client.chat.completions.create(
  model="gpt-4.1-nano",
  messages=[
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_message}
  ]
)


    response = completion.choices[0].message.content
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
