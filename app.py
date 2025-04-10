import os
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medicalbot"

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY or PINECONE_API_KEY in .env")

# Flask setup
app = Flask(__name__)
CORS(app)

# Gemini setup
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Pinecone setup
pc = PineconeClient(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [idx['name'] for idx in pc.list_indexes()]:
    raise ValueError(f"Index '{INDEX_NAME}' not found in Pinecone.")

# Load Pinecone Index and retriever
pinecone_index = pc.Index(INDEX_NAME)
retriever = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
)

# In-memory chat histories
chat_histories = {}

def get_chat_history(session_id):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

def generate_response_with_retry(prompt, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except genai.RateLimitError:
            print(f"Rate limit hit. Retrying... ({attempt + 1})")
            time.sleep(delay)
    return "Rate limit exceeded. Try again later."

def custom_rag_chain(query, session_id):
    history = get_chat_history(session_id)
    docs = retriever.similarity_search(query, k=4)
    past_messages = "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages])

    if docs:
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"Chat History:\n{past_messages}\n\nBased on the medical docs below, answer:\n\n{context}\n\nQuestion: {query}"
    else:
        prompt = f"Chat History:\n{past_messages}\n\nQuestion: {query}"

    response = generate_response_with_retry(prompt)
    history.add_user_message(query)
    history.add_ai_message(response)
    return response

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Medical Chatbot!"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message", "")
    session_id = data.get("session_id", "default")

    if not query:
        return jsonify({"error": "No message provided"}), 400

    response = custom_rag_chain(query, session_id)
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
