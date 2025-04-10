import os
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from pinecone import Pinecone, Index
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medicalbot"

# Check API keys
if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ Missing API keys! Check your .env file.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
existing_indexes = [index_info['name'] for index_info in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    raise ValueError(f"❌ Index '{INDEX_NAME}' not found in Pinecone. Check your Pinecone console.")

# Load the Pinecone index
pinecone_index = pc.Index(INDEX_NAME)

# Load Retriever
retriever = PineconeVectorStore(
    index=pinecone_index,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    text_key="text"  # Ensure this matches the metadata key in your Pinecone data
)
print("✅ Retriever loaded successfully!")

# In-memory chat history (Replace with persistent store if needed)
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
            print(f"⚠️ Rate limit reached. Retrying in {delay} seconds... (Attempt {attempt + 1})")
            time.sleep(delay)
    return "❌ Error: Rate limit exceeded. Please try again later."

def custom_rag_chain(query, session_id):
    history = get_chat_history(session_id)
    retrieved_docs = retriever.as_retriever().invoke(query)
    past_messages = "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages])

    if retrieved_docs:
        retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"Chat History:\n{past_messages}\n\nBased on the following medical references, answer the question:\n\n{retrieved_text}\n\nQuestion: {query}"
    else:
        print("⚠️ No relevant documents found. Answering from general knowledge...")
        prompt = f"Chat History:\n{past_messages}\n\nQuestion: {query}"

    response = generate_response_with_retry(prompt)

    # Save to history
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
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
