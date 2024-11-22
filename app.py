import time
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
from termcolor import colored
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import easyocr
import numpy as np
import fitz  # PyMuPDF for extracting images from PDFs
from PIL import Image, ImageOps
import io
import base64
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import os
from dotenv import load_dotenv
from threading import Lock

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flags and locks
is_ready = False
is_initializing = False
startup_time = datetime.now()
init_lock = Lock()

# Load environment variables
load_dotenv()
API_KEY = os.getenv("MY_API_KEY")  # Set your Gemini API key in .env file
genai.configure(api_key=API_KEY)

# Initialize OCR and embedding model
reader = None
embedding_model = None
vectorstore = None
images_with_context = None

# Utility Functions
def load_pdf_text(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in pdf_reader.pages]

def contains_kavach_content(image):
    image_np = np.array(image)
    results = reader.readtext(image_np, detail=0)
    if not results:
        return False
    detected_text = " ".join(results).lower()
    kavach_keywords = ["speed", "limit", "display", "signal", "distance",
                       "authority", "brake", "train", "kavach", "dmi"]
    return any(keyword in detected_text for keyword in kavach_keywords)

def is_relevant_symbol(image):
    grayscale = ImageOps.grayscale(image)
    avg_brightness = np.mean(np.array(grayscale))
    return 30 < avg_brightness < 200

def is_relevant_image(query, image_context):
    query_keywords = ['color', 'speed', 'permissible', 'brakes', 'train', 'signal', 'limit', 'display']
    image_text = image_context["context"]["page_text"].lower()
    if any(keyword in image_text for keyword in query_keywords):
        return True
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    context_embedding = embedding_model.encode(image_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(query_embedding, context_embedding).item()
    return similarity > 0.6

def extract_images_with_context(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images_with_context = []

    for page_num in range(pdf_document.page_count):
        if page_num == 0:
            continue
        page = pdf_document.load_page(page_num)
        page_text = page.get_text("text").strip()
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"]))
            if contains_kavach_content(image) or is_relevant_symbol(image):
                image_base64 = image_to_base64(image)
                images_with_context.append({
                    "image_data": image_base64,
                    "context": {
                        "page_num": page_num + 1,
                        "page_text": page_text
                    }
                })
    return images_with_context

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def create_chunks(pdf_pages):
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for i, page_text in enumerate(pdf_pages):
        for chunk in text_splitter.split_text(page_text):
            chunks.append({"text": chunk, "page_num": i + 1})
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings()
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [{'page_num': chunk['page_num']} for chunk in chunks]
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

def generate_kavach_response(context_text, query):
    prompt = f"""
    Context:
    {context_text}

    Question:
    {query}

    Please provide a clear and concise answer based on the context above.
    """
    model = genai.GenerativeModel("gemini-1.5-pro-002")
    response = model.generate_content(prompt)
    return response.text.strip()

def match_query_to_images(query, relevant_pages, images_with_context):
    matched_images = []
    extended_pages = set(relevant_pages)
    for page in relevant_pages:
        extended_pages.update({page - 2, page + 1})
    for img_context in images_with_context:
        if img_context["context"]["page_num"] in extended_pages and is_relevant_image(query, img_context):
            matched_images.append(img_context)
    return matched_images

# Initialization Logic
def initialize_dependencies():
    global reader, embedding_model, vectorstore, images_with_context, is_ready, is_initializing
    with init_lock:
        logger.info("Starting initialization...")
        is_initializing = True
        is_ready = False
        try:
            # Initialize OCR and embedding model
            reader = easyocr.Reader(['en'])
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Load PDF and process
            pdf_pages = load_pdf_text("Annexure-B.pdf")
            images_with_context = extract_images_with_context("Annexure-B.pdf")
            kavach_chunks = create_chunks(pdf_pages)
            vectorstore = create_vector_store(kavach_chunks)

            logger.info("Initialization complete. Application is ready.")
            is_ready = True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            is_ready = False
        finally:
            is_initializing = False

def start_initialization():
    thread = threading.Thread(target=initialize_dependencies)
    thread.start()

# Endpoints
@app.route("/ping", methods=["GET"])
def ping():
    logger.info(f"is_initializing: {is_initializing}, is_ready: {is_ready}")
    if is_initializing:
        return jsonify({"message": "Initializing, please wait...", "status": "Initializing"}), 503
    elif is_ready:
        uptime = datetime.now() - startup_time
        return jsonify({"message": "Pong!", "status": "Ready", "uptime": str(uptime)}), 200
    else:
        return jsonify({"message": "Not initialized. Please activate first.", "status": "Not Ready"}), 503

@app.route("/activate", methods=["POST"])
def activate():
    global is_ready, is_initializing
    data = request.get_json()
    if not data or data.get("number") != "1505":
        return jsonify({"error": "Invalid activation request"}), 400

    if is_initializing:
        return jsonify({"message": "Already initializing, please wait..."}), 200

    logger.info("Reinitialization triggered by activation request.")
    start_initialization()
    return jsonify({"message": "Initialization started"}), 200

@app.route("/query", methods=["POST"])
def query_kavach():
    if not is_ready:
        return jsonify({"error": "Service is initializing, please try again later."}), 503

    data = request.get_json()
    query = data.get("query", "").lower()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(query)
    combined_text = "\n".join([doc.page_content for doc in relevant_docs])
    relevant_pages = [doc.metadata['page_num'] for doc in relevant_docs]

    matched_images = match_query_to_images(query, relevant_pages, images_with_context)
    response_text = generate_kavach_response(combined_text, query) if combined_text.strip() else "No relevant information found."

    images = [{"page_num": img["context"]["page_num"], "image_data": img["image_data"]} for img in matched_images]
    return jsonify({"response": response_text, "images": images})

# Start Initialization and Run App
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 8000)))
