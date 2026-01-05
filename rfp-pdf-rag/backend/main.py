import os
import nltk
# --- PRE-REQUISITES & PATHS ---
import pytesseract


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- TOOLS TO BUILD THE AI BRAIN ---
# HumanMessage: How the AI sees our text
from langchain_core.messages import HumanMessage
# ChatPromptTemplate: The "Instructions" we give the AI on how to behave
from langchain_core.prompts import ChatPromptTemplate
# OllamaEmbeddings: Converts text into numbers (vectors) so the AI can "calculate" meaning
from langchain_ollama import OllamaEmbeddings
# ChatOllama: The actual Large Language Model (Llama 3) that talks to us
from langchain_ollama import ChatOllama
# Chroma: The "Vector Database" (The long-term memory storage)
from langchain_chroma import Chroma
# UnstructuredLoader: The "Eyes" that read PDFs, detect tables, and see images
from langchain_unstructured import UnstructuredLoader

# --- TOOLS FOR SEARCHING & INDEXING ---
# index/SQLRecordManager: Prevents the AI from reading the same file twice (Incremental Sync)
from langchain_classic.indexes import SQLRecordManager, index
# EnsembleRetriever: Combines "Keyword Search" with "AI Meaning Search"
from langchain_classic.retrievers import EnsembleRetriever
# BM25Retriever: Traditional "Keyword Search" (like Ctrl+F)
from langchain_community.retrievers import BM25Retriever
# Chains: The "Logic Flow" that connects searching to answering
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# --- PRE-REQUISITES & PATHS ---
# Tell the text-processor where the English language grammar files are
nltk.data.path.append(r"C:\nltk_data")

# Tell the PDF-reader where the conversion tool is located
POPPLER_PATH = r"C:\Softwares\poppler\Library\bin"

# 1. Point to the actual program file
pytesseract.pytesseract.tesseract_cmd = r"C:\Softwares\tesseract\tesseract.exe"

# 2. Tell the system where the 'tessdata' folder is located
# Note: Tesseract often prefers the path TO the tessdata folder itself
os.environ['TESSDATA_PREFIX'] = r"C:\Softwares\tesseract\tessdata"

# 3. Add the tesseract folder to the Windows PATH temporarily so 'unstructured' can see it
os.environ['PATH'] += os.pathsep + r"C:\Softwares\tesseract"



# --- FOLDER SETTINGS ---
# Where your RFP PDF files are stored
RFP_FOLDER = "C:\\learn-AI\\RFP-PDF-RAG\\rfp_docs\\"
# Where the AI saves its "meaning-based" memory
CHROMA_DIR = "C:\\learn-AI\\RFP-PDF-RAG\\chroma_db_store"
# A small database that remembers which files were already processed
RECORD_DB = f"sqlite:///C:/learn-AI/RFP-PDF-RAG/record_manager.db"

# --- MODEL SELECTION ---
# The model used for answering questions
TEXT_MODEL = "llama3:8b-instruct-q4_K_M" 
# The model used to "see" and describe diagrams/images in the PDF
VISION_MODEL = "llama3.2-vision"
# The model that turns text into math-vectors
EMBEDDING_MODEL = "nomic-embed-text"

# Global holder for our search engine so it can be used across the whole app
hybrid_retriever = None

# FUNCTION: How the AI describes a picture it finds in your RFP
def describe_image_with_vision(base64_image):
    try:
        # Load the Vision AI
        vision_llm = ChatOllama(model=VISION_MODEL)
        # Ask the AI to look at the image data and explain it
        msg = HumanMessage(content=[
            {"type": "text", "text": "Describe this RFP diagram/chart in detail for text-based retrieval."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])
        res = vision_llm.invoke([msg])
        return res.content
    except Exception as e:
        print(f"⚠️ Vision Error: {e}")
        return "Visual data detected."

# LIFESPAN: What the program does the moment it starts up
@asynccontextmanager
async def lifespan(app: FastAPI):
    global hybrid_retriever
    # Initialize the "Meaning-to-Math" converter
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 1. Open the existing Memory Store (Chroma)
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="rfp_expert_v3"
    )

    # 2. Open the "Record Manager" (The notebook that tracks file changes)
    record_manager = SQLRecordManager("chroma/rfp_v3", db_url=RECORD_DB)
    # Create the record-keeping table if it doesn't exist
    record_manager.create_schema()

    # 3. Look inside your RFP folder for PDF files
    print(f"📂 Checking for updates in: {RFP_FOLDER}")
    pdf_files = [f for f in os.listdir(RFP_FOLDER) if f.endswith('.pdf')]
    
    all_docs = []
    # Loop through every PDF found
    for pdf in pdf_files:
        path = os.path.join(RFP_FOLDER, pdf)
        # Use the "High-Resolution" reader to find tables and images
        loader = UnstructuredLoader(
            file_path=path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            chunking_strategy="by_title",
            max_characters=1000,
            poppler_path=POPPLER_PATH
        )
        
        # Read the pages and clean up the hidden data (metadata)
        elements = loader.load()
        for doc in elements:
            # If the reader found an image, use the Vision AI to describe it
            if doc.metadata.get("category") == "Image" and "image_base64" in doc.metadata:
                doc.page_content = f"[VISUAL DATA]: {describe_image_with_vision(doc.metadata['image_base64'])}"
            
            # Chroma database cannot handle "lists" in metadata, so we convert them to strings
            clean_meta = {}
            for k, v in doc.metadata.items():
                # If the value is a list (like ['eng']), join it into a string ('eng')
                clean_meta[k] = ", ".join(map(str, v)) if isinstance(v, list) else v
            doc.metadata = clean_meta
            all_docs.append(doc)

    # 4. Sync the data (Only process new files, skip the ones already known)
    if all_docs:
        print("🧠 Syncing Vector Database (Skipping existing files)...")
        # The 'index' function compares your folder to the Record Manager
        sync_stats = index(
            all_docs,
            record_manager,
            vector_store,
            cleanup="incremental", # This avoids duplication
            source_id_key="source"
        )
        print(f"📊 Indexing Stats: {sync_stats}")
    else:
        print("ℹ️ No documents found to process.")

    # 5. Build the "Hybrid" Search Engine
    # Get all the text currently in the memory
    data = vector_store.get()
    current_texts = data['documents']
    
    # Create the "AI Meaning" searcher
    v_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # If we have documents, create the "Keyword" searcher
    if current_texts:
        b_retriever = BM25Retriever.from_texts(current_texts)
        # Combine both (30% weight to keywords, 70% to AI meaning)
        hybrid_retriever = EnsembleRetriever(
            retrievers=[b_retriever, v_retriever],
            weights=[0.3, 0.7]
        )
        print("🚀 RFP Intelligence System is ONLINE.")
    else:
        print("⚠️ System online but Memory is empty. Add PDFs to the folder.")

    yield

# Create the Web API application
app = FastAPI(lifespan=lifespan)

# Setup "CORS" (Allows your React/Frontend website to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define what a "Chat Request" looks like (Just a question string)
class QueryRequest(BaseModel):
    question: str

# THE CHAT ENDPOINT: Where you send your questions
@app.post("/chat")
async def chat(request: QueryRequest):
    # Error if the search engine hasn't finished loading
    if not hybrid_retriever:
        raise HTTPException(status_code=503, detail="System not ready.")

    # Instructions for the AI on how to handle RFP responses
    system_prompt = (
        "You are an expert RFP Assistant. Use the context provided to answer.\n"
        "1. If there are tables, explain the data clearly.\n"
        "2. Use [VISUAL DATA] descriptions to answer questions about diagrams.\n"
        "3. Always mention which PDF file you got the answer from.\n"
        "4. If you can't find the answer, say 'Information not found'.\n\n"
        "Context: {context}"
    )
    
    # Merge the instructions with the user's question
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Connect to the Llama 3 model
    llm = ChatOllama(model=TEXT_MODEL, temperature=0)
    # Combine the documents found during search into one answer
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    # Create the final "Search -> Think -> Answer" chain
    retrieval_chain = create_retrieval_chain(hybrid_retriever, combine_docs_chain)

    try:
        # Run the search and get the response
        response = retrieval_chain.invoke({"input": request.question})
        # Extract the names of the PDF files used for the answer
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in response['context']]))
        return {"answer": response["answer"], "sources": sources}
    except Exception as e:
        # If something crashes, return the error message
        raise HTTPException(status_code=500, detail=str(e))

# ROOT ENDPOINT: A simple "Hello" page to see if the server is up
@app.get("/")
async def root():
    return {"status": "Running", "db_location": CHROMA_DIR}

# START COMMAND: Run the server on Port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)