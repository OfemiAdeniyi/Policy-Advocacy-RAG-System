import os
import time
from llama_index.readers.llama_parse import LlamaParse
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION ---
# Replace with your actual Llama Cloud API Key
os.environ["LLAMA_CLOUD_API_KEY"] = "API KEY"

# Update these paths as needed
folder_path = r"C:\Users\USER\Documents\FLAMONG\Policy Project\PolicyDocs"
persist_db_path = "./policy_db_local"
progress_file = "processed_files.txt"

# --- 2. INITIALIZE COMPONENTS ---
parser = LlamaParse(result_type="markdown", verbose=True)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load progress from the bookmark file
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        processed_files = set(f.read().splitlines())
else:
    processed_files = set()

# Load/Initialize the Vector Database
vector_db = Chroma(persist_directory=persist_db_path, embedding_function=embeddings)

def save_bookmark(filename):
    with open(progress_file, "a") as f:
        f.write(filename + "\n")

# --- 3. MAIN INGESTION LOOP ---
print(f"--- Starting Ingestion for Policy.ng ---")
print(f"Status: {len(processed_files)} files already indexed. Remaining to check...")

all_pdfs = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

for filename in all_pdfs:
    if filename in processed_files:
        continue # Skip already finished files
    
    file_path = os.path.join(folder_path, filename)
    print(f"\n[TASK] Processing: {filename}")
    
    doc_content = []
    
    # STEP A: Try Local Load (Free)
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        # Verify if there's actual text (not a scan)
        text_sample = "".join([p.page_content for p in pages]).strip()
        
        if len(text_sample) > 200:
            print(f"-> Success: Digital text detected.")
            doc_content = pages
        else:
            print(f"-> Scan detected. Routing to LlamaParse...")
            raise Exception("Scan detected")
            
    except Exception:
        # STEP B: Cloud Recovery (LlamaParse)
        try:
            llama_docs = parser.load_data(file_path)
            # Convert LlamaIndex docs to LangChain docs
            doc_content = [Document(page_content=l.text, metadata={"source": filename, "method": "llamaparse"}) for l in llama_docs]
            print(f"-> Success: Document recovered via Cloud OCR.")
        except Exception as e:
            print(f"!! CRITICAL ERROR on {filename}: {e}")
            continue

    # STEP C: Manual Batching & Database Save
    if doc_content:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(doc_content)
        
        # MANUAL BATCHING: To fix the 'Batch size greater than 5461' error
        safe_batch_size = 2000
        total_chunks = len(chunks)
        print(f"-> Indexing {total_chunks} chunks in safe batches...")
        
        try:
            for i in range(0, total_chunks, safe_batch_size):
                batch = chunks[i : i + safe_batch_size]
                vector_db.add_documents(batch)
                print(f"   Indexed batch {i//safe_batch_size + 1}...")
            
            # Save progress ONLY after all chunks for this file are saved
            save_bookmark(filename)
            processed_files.add(filename)
            print(f"DONE: {filename} is now in the database.")
            
        except Exception as e:
            print(f"!! DATABASE ERROR while saving {filename}: {e}")
            # We don't save the bookmark, so the script will retry this file next time

print(f"\n--- SESSION COMPLETE ---")
print(f"Total files now in index: {len(processed_files)}")
