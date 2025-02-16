import os
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medicalbot" 

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ Missing API keys! Check your .env file.")


app = Flask(__name__)
CORS(app)  

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")  


pc = Pinecone(api_key=PINECONE_API_KEY)


if INDEX_NAME not in [index_info['name'] for index_info in pc.list_indexes()]:
    raise ValueError(f"❌ Index '{INDEX_NAME}' not found in Pinecone. Check your Pinecone console.")

retriever = PineconeVectorStore.from_existing_index(
    INDEX_NAME,
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
print("✅ Retriever loaded successfully!")

def generate_response_with_retry(prompt, max_retries=3, delay=5):
    """Handles Gemini AI API rate limits with retry logic."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text 
        except genai.RateLimitError:
            print(f"⚠️ Rate limit reached. Retrying in {delay} seconds... (Attempt {attempt + 1})")
            time.sleep(delay)
    return "❌ Error: Rate limit exceeded. Please try again later."

def custom_rag_chain(query):
    """Retrieves relevant documents and generates an answer."""
    retrieved_docs = retriever.as_retriever().invoke(query)  

    if retrieved_docs:  
        retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"Based on the following medical references, answer the question:\n\n{retrieved_text}\n\nQuestion: {query}"
        return generate_response_with_retry(prompt)
    
    else:  
        print("⚠️ No relevant documents found. Answering from general knowledge...")
        return generate_response_with_retry(query)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message", "")

    if not query:
        return jsonify({"error": "No message provided"}), 400

    response = custom_rag_chain(query)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
